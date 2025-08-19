from neo4j import GraphDatabase
import re
from typing import List, Dict, Tuple
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
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
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            num_predict=MODEL_NUM_PREDICT,
            top_p=MODEL_TOP_P,
            top_k=MODEL_TOP_K
        )
        
        # 創建自定義的 Cypher prompt
        custom_cypher_prompt = PromptTemplate.from_template("""
Task: 根據用戶問題生成Cypher查詢語句

Schema:
- 節點: Entity (屬性: name)
- 關係: RELATION (屬性: name, source)

Rules:
1. 只返回Cypher查詢語句，不要其他說明文字
2. 使用 CONTAINS 和 toLower() 進行模糊匹配
3. 提取用戶問題中的關鍵詞進行查詢
4. 限制返回結果數量 LIMIT 20

Query Templates:
- 單一關鍵詞: MATCH (s:Entity)-[r:RELATION]->(o:Entity) WHERE toLower(s.name) CONTAINS toLower("keyword") OR toLower(o.name) CONTAINS toLower("keyword") RETURN s.name as subject, r.name as predicate, o.name as object LIMIT 20
- 多關鍵詞: MATCH (s:Entity)-[r:RELATION]->(o:Entity) WHERE toLower(s.name) CONTAINS toLower("keyword1") OR toLower(o.name) CONTAINS toLower("keyword1") RETURN s.name as subject, r.name as predicate, o.name as object LIMIT 20

Question: {question}

Cypher:""")

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