from neo4j import GraphDatabase
import re
from typing import List, Dict, Tuple
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    MODEL_TEMPERATURE,
    MODEL_NUM_PREDICT,
    MODEL_TOP_P,
    MODEL_TOP_K,
    CYPHER_GENERATION_PROMPT,
    KNOWLEDGE_GRAPH_QA_PROMPT
)

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
        custom_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_PROMPT)

        # 創建自定義的 QA prompt
        custom_qa_prompt = PromptTemplate.from_template(KNOWLEDGE_GRAPH_QA_PROMPT)

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

    def hybrid_search_context_only(self, query: str) -> Dict:
        """
        僅進行知識圖譜檢索,不生成LLM回答
        用於hybrid-all模式,避免重複生成回答
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

            return {
                'cypher_query': cypher_query,
                'context': context_data,
                'mode': 'context_only'
            }
        except Exception as e:
            print(f"❌ 知識圖譜檢索錯誤: {e}")
            import traceback
            traceback.print_exc()
            return {
                'cypher_query': '',
                'context': [],
                'mode': 'context_only'
            }

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