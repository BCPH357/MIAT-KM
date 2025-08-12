from neo4j import GraphDatabase
import re
from typing import List, Dict, Tuple
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.prompts.prompt import PromptTemplate
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MODEL_TEMPERATURE, MODEL_NUM_PREDICT, MODEL_TOP_P, MODEL_TOP_K

class Neo4jKnowledgeRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 初始化 LangChain 組件
        self.graph = Neo4jGraph(
            url=uri,
            username=user,
            password=password
        )
        
        # 初始化 Ollama LLM
        self.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            num_predict=MODEL_NUM_PREDICT,
            top_p=MODEL_TOP_P,
            top_k=MODEL_TOP_K
        )
        
        # 創建自定義的 Cypher prompt
        custom_cypher_prompt = PromptTemplate.from_template("""
你是圖資料庫的專家，根據用戶的問題寫一個 Cypher 查詢來查詢知識圖譜。

重要注意事項：
1. 圖中的實體名稱可能含有空格（例如："MIAT 方法論"），請使用模糊比對避免空格造成錯誤
2. 優先使用 CONTAINS 進行模糊匹配，而不是精確的 = 匹配
3. 使用 toLower() 進行不區分大小寫的查詢
4. 如果查詢多個關鍵詞，可以將它們分開查詢

圖結構：
- 節點標籤：Entity (屬性: name)
- 關係類型：RELATION (屬性: name, source)

查詢模式範例：
1. 單一實體查詢：
   MATCH (s:Entity)-[r:RELATION]->(o:Entity)
   WHERE toLower(s.name) CONTAINS toLower("關鍵詞")
   RETURN s.name as subject, r.name as predicate, o.name as object

2. 多關鍵詞查詢：
   MATCH (s:Entity)-[r:RELATION]->(o:Entity)
   WHERE toLower(s.name) CONTAINS toLower("關鍵詞1") AND toLower(s.name) CONTAINS toLower("關鍵詞2")
   RETURN s.name as subject, r.name as predicate, o.name as object

3. 雙向查詢（主語或賓語）：
   MATCH (s:Entity)-[r:RELATION]->(o:Entity)
   WHERE toLower(s.name) CONTAINS toLower("關鍵詞") OR toLower(o.name) CONTAINS toLower("關鍵詞")
   RETURN s.name as subject, r.name as predicate, o.name as object

用戶問題: {question}

請生成適當的 Cypher 查詢：
""")

        # 創建自定義的 QA prompt
        custom_qa_prompt = PromptTemplate.from_template("""
你是一個專業的知識問答助手。請基於以下從知識圖譜檢索到的信息來詳細回答用戶的問題。

從知識圖譜檢索到的相關信息：
{context}

用戶問題：{question}

請根據上述檢索到的知識信息，提供一個詳細、完整且有條理的回答。要求：

1. **完整性**：盡可能整合所有相關的檢索信息
2. **詳細性**：提供豐富的細節和背景信息
3. **結構化**：使用清晰的段落和邏輯結構
4. **準確性**：嚴格基於檢索到的知識，不要添加不存在的信息
5. **關聯性**：解釋不同信息之間的關係和聯繫

如果檢索到的信息不足以完全回答問題，請明確說明哪些方面的信息不足，並基於已有信息提供盡可能詳細的回答。

請開始你的詳細回答：
""")

        # 創建 GraphCypherQAChain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=custom_cypher_prompt,
            qa_prompt=custom_qa_prompt,
            verbose=True,
            return_intermediate_steps=True,
            return_direct=False,
            allow_dangerous_requests=True
        )
    
    def close(self):
        self.driver.close()
    
    def hybrid_search(self, query: str, ollama_client) -> Dict:
        """
        混合RAG模式：使用LangChain檢索 + 自定義回答生成
        """
        try:
            # 使用LangChain僅進行Cypher查詢和數據檢索
            result = self.cypher_chain.invoke({"query": query})
            
            # 解析檢索結果
            cypher_query = ''
            context_data = []
            
            if 'intermediate_steps' in result and result['intermediate_steps']:
                if len(result['intermediate_steps']) > 0:
                    cypher_query = result['intermediate_steps'][0].get('query', '')
                if len(result['intermediate_steps']) > 1:
                    context_data = result['intermediate_steps'][1].get('context', [])
            
            # 格式化檢索到的知識用於自定義生成
            if context_data:
                knowledge_items = []
                for i, item in enumerate(context_data, 1):
                    if isinstance(item, dict):
                        subject = item.get('subject', '')
                        predicate = item.get('predicate', '')
                        object_val = item.get('object', '')
                        knowledge_items.append(f"{i}. {subject} → [{predicate}] → {object_val}")
                    else:
                        knowledge_items.append(f"{i}. {item}")
                
                knowledge_context = "從知識圖譜檢索到的相關信息：\n" + "\n".join(knowledge_items)
                
                # 使用自定義RAG生成詳細回答
                detailed_answer = ollama_client.rag_generate(
                    model=OLLAMA_MODEL,
                    user_query=query,
                    knowledge_context=knowledge_context,
                    temperature=0.7
                )
            else:
                detailed_answer = "很抱歉，知識庫中沒有找到相關信息來回答您的問題。"
            
            print(f"🔍 混合模式完整結果結構: {result.keys()}")
            if 'intermediate_steps' in result:
                print(f"📊 中間步驟數量: {len(result['intermediate_steps'])}")
                for i, step in enumerate(result['intermediate_steps']):
                    print(f"   步驟 {i}: {step.keys()}")
            
            return {
                'answer': detailed_answer,
                'cypher_query': cypher_query,
                'context': context_data,
                'full_result': result,
                'raw_intermediate_steps': result.get('intermediate_steps', []),
                'mode': 'hybrid'
            }
        except Exception as e:
            print(f"❌ 混合模式查詢錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f'查詢過程中發生錯誤: {e}',
                'cypher_query': '',
                'context': [],
                'full_result': {},
                'raw_intermediate_steps': [],
                'mode': 'hybrid'
            }

    def langchain_search(self, query: str) -> Dict:
        """
        使用 LangChain GraphCypherQAChain 進行智能查詢
        """
        try:
            result = self.cypher_chain.invoke({"query": query})
            
            # 解析 intermediate_steps
            cypher_query = ''
            context_data = []
            
            if 'intermediate_steps' in result and result['intermediate_steps']:
                # 第一個步驟包含 Cypher 查詢
                if len(result['intermediate_steps']) > 0:
                    cypher_query = result['intermediate_steps'][0].get('query', '')
                
                # 第二個步驟包含檢索到的數據
                if len(result['intermediate_steps']) > 1:
                    context_data = result['intermediate_steps'][1].get('context', [])
            
            # 從 full_result 中獲取額外信息用於調試
            print(f"🔍 LangChain 完整結果結構: {result.keys()}")
            if 'intermediate_steps' in result:
                print(f"📊 中間步驟數量: {len(result['intermediate_steps'])}")
                for i, step in enumerate(result['intermediate_steps']):
                    print(f"   步驟 {i}: {step.keys()}")
            
            return {
                'answer': result.get('result', ''),
                'cypher_query': cypher_query,
                'context': context_data,
                'full_result': result,
                'raw_intermediate_steps': result.get('intermediate_steps', [])
            }
        except Exception as e:
            print(f"❌ LangChain 查詢錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f'查詢過程中發生錯誤: {e}',
                'cypher_query': '',
                'context': [],
                'full_result': {},
                'raw_intermediate_steps': []
            }
    

    

    

    

    

    
 