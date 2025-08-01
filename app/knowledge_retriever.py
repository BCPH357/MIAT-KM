from neo4j import GraphDatabase
import re
from typing import List, Dict, Tuple
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama
from langchain.prompts.prompt import PromptTemplate

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
            model="gemma3:12b",
            base_url="http://ollama:11434"
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

        # 創建 GraphCypherQAChain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=custom_cypher_prompt,
            verbose=True,
            return_intermediate_steps=True,
            return_direct=False,
            allow_dangerous_requests=True
        )
    
    def close(self):
        self.driver.close()
    
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
    

    

    

    

    

    
 