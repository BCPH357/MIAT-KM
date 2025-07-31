from knowledge_retriever import Neo4jKnowledgeRetriever
from ollama_client import OllamaClient
import time
from typing import Dict, Any

class RAGSystem:
    def __init__(self, 
                 neo4j_uri: str = "bolt://neo4j:7687",
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password123",
                 ollama_url: str = "http://ollama:11434",
                 model_name: str = "gemma3:12b"):
        
        self.knowledge_retriever = Neo4jKnowledgeRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.ollama_client = OllamaClient(ollama_url)
        self.model_name = model_name
        
        # 檢查模型是否可用
        if not self.ollama_client.check_model_available(model_name):
            print(f"警告: 模型 {model_name} 不可用")
            available_models = self.ollama_client.list_models()
            if not available_models.get("error"):
                print("可用模型:", [m["name"] for m in available_models.get("models", [])])
    
    def answer_question(self, user_query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        回答用戶問題
        """
        start_time = time.time()
        
        result = {
            "query": user_query,
            "use_rag": use_rag,
            "knowledge_context": "",
            "answer": "",
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0,
            "knowledge_items_count": 0
        }
        
        if use_rag:
            # 步驟 1: 從 Neo4j 檢索相關知識
            retrieval_start = time.time()
            knowledge_context = self.knowledge_retriever.comprehensive_search(user_query)
            retrieval_time = time.time() - retrieval_start
            
            result["knowledge_context"] = knowledge_context
            result["retrieval_time"] = retrieval_time
            
            # 計算知識項目數量
            if "沒有找到相關的知識" not in knowledge_context:
                # 簡單計算知識項目數量（以編號開始的行）
                lines = knowledge_context.split('\n')
                knowledge_items = [line for line in lines if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 20))]
                result["knowledge_items_count"] = len(knowledge_items)
            
            # 步驟 2: 使用 RAG 生成回答
            generation_start = time.time()
            answer = self.ollama_client.rag_generate(
                model=self.model_name,
                user_query=user_query,
                knowledge_context=knowledge_context
            )
            generation_time = time.time() - generation_start
            
            result["answer"] = answer
            result["generation_time"] = generation_time
        
        else:
            # 不使用 RAG，直接生成回答
            generation_start = time.time()
            answer = self.ollama_client.simple_generate(
                model=self.model_name,
                user_query=user_query
            )
            generation_time = time.time() - generation_start
            
            result["answer"] = answer
            result["generation_time"] = generation_time
        
        result["total_time"] = time.time() - start_time
        return result
    
    def compare_rag_vs_normal(self, user_query: str) -> Dict[str, Any]:
        """
        比較使用 RAG 和不使用 RAG 的回答
        """
        print("正在生成 RAG 回答...")
        rag_result = self.answer_question(user_query, use_rag=True)
        
        print("正在生成普通回答...")
        normal_result = self.answer_question(user_query, use_rag=False)
        
        return {
            "query": user_query,
            "rag_result": rag_result,
            "normal_result": normal_result
        }
    
    def search_knowledge_only(self, query: str) -> str:
        """
        僅搜索知識，不生成回答
        """
        return self.knowledge_retriever.comprehensive_search(query)
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """
        獲取特定實體的詳細信息
        """
        relations = self.knowledge_retriever.search_by_entity(entity_name)
        neighbors = self.knowledge_retriever.get_entity_neighbors(entity_name)
        
        return {
            "entity": entity_name,
            "relations": relations,
            "neighbors": neighbors
        }
    
    def interactive_qa(self):
        """
        互動式問答
        """
        print("=== RAG 知識問答系統 ===")
        print("直接輸入問題，系統會自動搜索知識庫並回答")
        print("輸入 'quit' 或 'exit' 退出")
        print("輸入 'compare <問題>' 比較 RAG 和普通回答")
        print("輸入 'search <關鍵字>' 僅搜索知識")
        print("輸入 'entity <實體名>' 查看實體信息")
        print("輸入 'detail <問題>' 查看詳細的檢索過程")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n請輸入問題: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if user_input.startswith('compare '):
                    query = user_input[8:].strip()
                    if query:
                        result = self.compare_rag_vs_normal(query)
                        self._print_comparison_result(result)
                    else:
                        print("請提供要比較的問題")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        knowledge = self.search_knowledge_only(query)
                        print(f"\n搜索結果:\n{knowledge}")
                    else:
                        print("請提供搜索關鍵字")
                
                elif user_input.startswith('entity '):
                    entity = user_input[7:].strip()
                    if entity:
                        info = self.get_entity_info(entity)
                        self._print_entity_info(info)
                    else:
                        print("請提供實體名稱")
                
                elif user_input.startswith('detail '):
                    query = user_input[7:].strip()
                    if query:
                        result = self.answer_question(query, use_rag=True)
                        self._print_rag_result(result)
                    else:
                        print("請提供要詳細分析的問題")
                
                elif user_input:
                    # 自動使用 RAG 回答問題
                    result = self.answer_question(user_input, use_rag=True)
                    self._print_simple_answer(result)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"發生錯誤: {e}")
        
        print("\n感謝使用 RAG 問答系統！")
    
    def _print_rag_result(self, result: Dict[str, Any]):
        """
        格式化打印 RAG 結果
        """
        print(f"\n{'='*50}")
        print(f"問題: {result['query']}")
        print(f"{'='*50}")
        
        if result['use_rag']:
            print(f"檢索到 {result['knowledge_items_count']} 個相關知識項目")
            print(f"檢索時間: {result['retrieval_time']:.2f} 秒")
            
            if result['knowledge_items_count'] > 0:
                print(f"\n檢索到的知識:")
                print(result['knowledge_context'])
        
        print(f"\n回答:")
        print(result['answer'])
        
        print(f"\n生成時間: {result['generation_time']:.2f} 秒")
        print(f"總時間: {result['total_time']:.2f} 秒")
    
    def _print_comparison_result(self, result: Dict[str, Any]):
        """
        格式化打印比較結果
        """
        print(f"\n{'='*50}")
        print(f"問題: {result['query']}")
        print(f"{'='*50}")
        
        print(f"\n【RAG 回答】(檢索到 {result['rag_result']['knowledge_items_count']} 個知識項目)")
        print(result['rag_result']['answer'])
        print(f"時間: {result['rag_result']['total_time']:.2f} 秒")
        
        print(f"\n【普通回答】")
        print(result['normal_result']['answer'])
        print(f"時間: {result['normal_result']['total_time']:.2f} 秒")
    
    def _print_entity_info(self, info: Dict[str, Any]):
        """
        格式化打印實體信息
        """
        print(f"\n實體: {info['entity']}")
        print("-" * 30)
        
        if info['relations']:
            print("關係:")
            for i, rel in enumerate(info['relations'], 1):
                direction = "→" if rel['direction'] == 'outgoing' else "←"
                print(f"  {i}. {rel['entity']} {direction} [{rel['relation']}] {direction} {rel['related_entity']}")
        else:
            print("沒有找到相關關係")
        
        if info['neighbors']:
            print("\n鄰居節點:")
            for i, neighbor in enumerate(info['neighbors'], 1):
                print(f"  {i}. {neighbor['neighbor']} (關係: {neighbor['relation']})")
    
    def _print_simple_answer(self, result: Dict[str, Any]):
        """
        簡化的答案顯示（只顯示答案，不顯示檢索詳情）
        """
        print(f"\n問題: {result['query']}")
        print("-" * 50)
        
        if result['use_rag'] and result['knowledge_items_count'] > 0:
            print(f"✓ 已從知識庫檢索到 {result['knowledge_items_count']} 個相關信息")
        
        print(f"\n回答:")
        print(result['answer'])
        
        if result['use_rag'] and result['knowledge_items_count'] == 0:
            print("\n(注意: 知識庫中沒有找到相關信息，以上為模型的一般知識回答)")
    
    def close(self):
        """
        關閉連接
        """
        self.knowledge_retriever.close() 