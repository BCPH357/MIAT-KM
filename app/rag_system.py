from knowledge_retriever import Neo4jKnowledgeRetriever
from ollama_client import OllamaClient
from vector_rag_processor import VectorRAGProcessor
import time
from typing import Dict, Any
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_BASE_URL, OLLAMA_MODEL

class RAGSystem:
    def __init__(self, 
                 neo4j_uri: str = NEO4J_URI,
                 neo4j_user: str = NEO4J_USER, 
                 neo4j_password: str = NEO4J_PASSWORD,
                 ollama_url: str = OLLAMA_BASE_URL,
                 model_name: str = OLLAMA_MODEL):
        
        self.knowledge_retriever = Neo4jKnowledgeRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.ollama_client = OllamaClient(ollama_url)
        self.model_name = model_name
        
        # 初始化Vector RAG處理器
        try:
            self.vector_rag_processor = VectorRAGProcessor()
            self.vector_available = True
        except Exception as e:
            print(f"警告: Vector RAG初始化失敗: {e}")
            self.vector_rag_processor = None
            self.vector_available = False
        
        # 檢查模型是否可用
        if not self.ollama_client.check_model_available(model_name):
            print(f"警告: 模型 {model_name} 不可用")
            available_models = self.ollama_client.list_models()
            if not available_models.get("error"):
                print("可用模型:", [m["name"] for m in available_models.get("models", [])])
    
    def answer_question(self, user_query: str, use_rag: bool = True, use_langchain: bool = False, use_hybrid: bool = False, use_vector: bool = False, use_hybrid_all: bool = False) -> Dict[str, Any]:
        """
        回答用戶問題
        """
        start_time = time.time()
        
        result = {
            "query": user_query,
            "use_rag": use_rag,
            "use_langchain": use_langchain,
            "use_hybrid": use_hybrid,
            "use_vector": use_vector,
            "use_hybrid_all": use_hybrid_all,
            "knowledge_context": "",
            "answer": "",
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0,
            "knowledge_items_count": 0,
            "cypher_query": "",
            "context_data": [],
            "vector_results": []
        }
        
        if use_hybrid_all:
            # 使用混合檢索模式：同時使用知識圖譜和向量檢索
            retrieval_start = time.time()
            
            # 獲取知識圖譜結果
            hybrid_result = self.knowledge_retriever.hybrid_search(user_query, self.ollama_client)
            
            # 獲取向量檢索結果
            vector_results = []
            if self.vector_available:
                try:
                    vector_search = self.vector_rag_processor.search_documents(user_query, n_results=5)
                    vector_results = vector_search['results']
                except Exception as e:
                    print(f"向量檢索失敗: {e}")
            
            retrieval_time = time.time() - retrieval_start
            
            # 組合知識上下文
            combined_context = f"Cypher查詢: {hybrid_result['cypher_query']}\n檢索到的圖譜數據: {hybrid_result['context']}\n"
            if vector_results:
                combined_context += "\n向量檢索結果:\n"
                for i, vr in enumerate(vector_results, 1):
                    combined_context += f"{i}. {vr['content'][:200]}... (來源: {vr['metadata'].get('source_file', 'Unknown')})\n"
            
            # 使用組合上下文生成回答
            generation_start = time.time()
            detailed_answer = self.ollama_client.rag_generate(
                model=self.model_name,
                user_query=user_query,
                knowledge_context=combined_context,
                temperature=0.7
            )
            generation_time = time.time() - generation_start
            
            result["answer"] = detailed_answer
            result["cypher_query"] = hybrid_result["cypher_query"]
            result["context_data"] = hybrid_result["context"]
            result["vector_results"] = vector_results
            result["knowledge_context"] = combined_context
            result["retrieval_time"] = retrieval_time
            result["generation_time"] = generation_time
            result["knowledge_items_count"] = len(hybrid_result["context"]) + len(vector_results)
            
        elif use_vector:
            # 純向量RAG模式
            if not self.vector_available:
                result["answer"] = "向量RAG系統不可用，請檢查配置"
                return result
            
            retrieval_start = time.time()
            vector_search = self.vector_rag_processor.search_documents(user_query, n_results=5)
            retrieval_time = time.time() - retrieval_start
            
            vector_results = vector_search['results']
            
            if vector_results:
                # 組織向量檢索上下文
                vector_context = "從向量數據庫檢索到的相關信息:\n"
                for i, vr in enumerate(vector_results, 1):
                    vector_context += f"{i}. {vr['content']} (相似度: {vr['similarity_score']:.3f}, 來源: {vr['metadata'].get('source_file', 'Unknown')})\n"
                
                # 生成回答
                generation_start = time.time()
                answer = self.ollama_client.rag_generate(
                    model=self.model_name,
                    user_query=user_query,
                    knowledge_context=vector_context,
                    temperature=0.7
                )
                generation_time = time.time() - generation_start
                
                result["answer"] = answer
                result["knowledge_context"] = vector_context
                result["vector_results"] = vector_results
                result["knowledge_items_count"] = len(vector_results)
            else:
                result["answer"] = "向量數據庫中沒有找到相關信息"
            
            result["retrieval_time"] = retrieval_time
            result["generation_time"] = generation_time if 'generation_time' in locals() else 0
            
        elif use_hybrid:
            # 使用混合RAG模式：LangChain檢索 + 自定義生成
            retrieval_start = time.time()
            hybrid_result = self.knowledge_retriever.hybrid_search(user_query, self.ollama_client)
            retrieval_time = time.time() - retrieval_start
            
            result["answer"] = hybrid_result["answer"]
            result["cypher_query"] = hybrid_result["cypher_query"]
            result["context_data"] = hybrid_result["context"]
            result["knowledge_context"] = f"Cypher查詢: {hybrid_result['cypher_query']}\n檢索到的數據: {hybrid_result['context']}"
            result["retrieval_time"] = retrieval_time
            result["generation_time"] = 0  # 包含在retrieval_time中
            result["knowledge_items_count"] = len(hybrid_result["context"]) if isinstance(hybrid_result["context"], list) else 0
            
        elif use_langchain:
            # 使用 LangChain GraphCypherQAChain
            retrieval_start = time.time()
            langchain_result = self.knowledge_retriever.langchain_search(user_query)
            retrieval_time = time.time() - retrieval_start
            
            result["answer"] = langchain_result["answer"]
            result["cypher_query"] = langchain_result["cypher_query"]
            result["context_data"] = langchain_result["context"]
            result["knowledge_context"] = f"Cypher查詢: {langchain_result['cypher_query']}\n檢索到的數據: {langchain_result['context']}"
            result["retrieval_time"] = retrieval_time
            result["generation_time"] = 0  # LangChain 內部處理
            result["knowledge_items_count"] = len(langchain_result["context"]) if isinstance(langchain_result["context"], list) else 0
            
        elif use_rag:
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
    

    
    def interactive_qa(self):
        """
        互動式問答 - 簡化版，直接使用 LangChain
        """
        print("=== RAG 知識問答系統 (增強版) ===")
        print("💡 使用 LangChain + Ollama + Neo4j 知識圖譜 + Vector RAG")
        print("\n🔧 可用命令:")
        print("  直接輸入問題 → 使用改進的LangChain模式")
        print("  'hybrid <問題>' → 使用混合RAG模式(推薦)")
        print("  'vector <問題>' → 使用純向量RAG模式")
        print("  'hybrid-all <問題>' → 使用全混合模式(知識圖譜+向量)")
        print("  'langchain <問題>' → 使用原始LangChain模式")
        print("  'compare <問題>' → 比較多種模式")
        print("  'quit' 或 'exit' → 退出系統")
        
        if self.vector_available:
            vector_stats = self.vector_rag_processor.get_database_stats()
            print(f"\n📊 向量數據庫狀態: {vector_stats['total_chunks']} chunks, {vector_stats['unique_files']} 文件")
        else:
            print("\n⚠️  向量RAG不可用")
        
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\n請輸入問題: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                elif user_input.startswith('hybrid '):
                    # 使用混合RAG模式
                    query = user_input[7:].strip()
                    if query:
                        result = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid=True)
                        self._print_hybrid_result(result)
                    else:
                        print("請提供問題")
                
                elif user_input.startswith('langchain '):
                    # 使用原始LangChain模式
                    query = user_input[10:].strip()
                    if query:
                        result = self.answer_question(query, use_rag=False, use_langchain=True)
                        self._print_langchain_result(result)
                    else:
                        print("請提供問題")
                
                elif user_input.startswith('vector '):
                    # 使用純向量RAG模式
                    query = user_input[7:].strip()
                    if query:
                        if self.vector_available:
                            result = self.answer_question(query, use_rag=False, use_langchain=False, use_vector=True)
                            self._print_vector_result(result)
                        else:
                            print("向量RAG系統不可用")
                    else:
                        print("請提供問題")
                
                elif user_input.startswith('hybrid-all '):
                    # 使用全混合模式
                    query = user_input[11:].strip()
                    if query:
                        if self.vector_available:
                            result = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid_all=True)
                            self._print_hybrid_all_result(result)
                        else:
                            print("向量RAG系統不可用，將使用知識圖譜模式")
                            result = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid=True)
                            self._print_hybrid_result(result)
                    else:
                        print("請提供問題")
                
                elif user_input.startswith('compare '):
                    # 比較多種模式
                    query = user_input[8:].strip()
                    if query:
                        self._compare_all_modes(query)
                    else:
                        print("請提供問題")
                
                elif user_input:
                    # 直接使用改進的LangChain模式
                    result = self.answer_question(user_input, use_rag=False, use_langchain=True)
                    self._print_simple_langchain_answer(result)
                
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
    
    def _print_langchain_result(self, result: Dict[str, Any]):
        """
        格式化打印 LangChain 結果
        """
        print(f"\n{'='*60}")
        print(f"問題: {result['query']}")
        print(f"【使用 LangChain GraphCypherQAChain】")
        print(f"{'='*60}")
        
        if result['cypher_query']:
            print(f"\n🔧 LLM 生成的 Cypher 查詢:")
            # 清理 cypher 前綴
            clean_query = result['cypher_query'].replace('cypher\n', '').strip()
            print(f"   {clean_query}")
        
        if result['context_data']:
            print(f"\n📊 從 Neo4j 檢索到的知識 ({len(result['context_data'])} 項):")
            for i, item in enumerate(result['context_data'], 1):
                if isinstance(item, dict):
                    subject = item.get('subject', '')
                    predicate = item.get('predicate', '')
                    object_val = item.get('object', '')
                    print(f"   {i}. {subject} → [{predicate}] → {object_val}")
                else:
                    print(f"   {i}. {item}")
        else:
            print(f"\n📊 沒有檢索到相關知識")
        
        print(f"\n🤖 LLM 最終回答:")
        print(f"   {result['answer']}")
        
        print(f"\n⏱️ 執行時間:")
        print(f"   檢索時間: {result['retrieval_time']:.2f} 秒")
        print(f"   總時間: {result['total_time']:.2f} 秒")
        
    def _print_simple_langchain_answer(self, result: Dict[str, Any]):
        """
        簡潔版的 LangChain 結果顯示
        """
        print(f"\n🤖 回答:")
        print(f"   {result['answer']}")
        
        if result['context_data']:
            print(f"\n📚 基於 {len(result['context_data'])} 個知識項目 (執行時間: {result['total_time']:.2f}s)")
        else:
            print(f"\n⚠️  知識庫中沒有找到相關信息 (執行時間: {result['total_time']:.2f}s)")
    
    def _print_hybrid_result(self, result: Dict[str, Any]):
        """
        格式化打印混合RAG結果
        """
        print(f"\n{'='*60}")
        print(f"問題: {result['query']}")
        print(f"【混合RAG模式】LangChain檢索 + 自定義生成")
        print(f"{'='*60}")
        
        if result['cypher_query']:
            print(f"\n🔧 LLM 生成的 Cypher 查詢:")
            clean_query = result['cypher_query'].replace('cypher\n', '').strip()
            print(f"   {clean_query}")
        
        if result['context_data']:
            print(f"\n📊 從 Neo4j 檢索到的知識 ({len(result['context_data'])} 項):")
            for i, item in enumerate(result['context_data'], 1):
                if isinstance(item, dict):
                    subject = item.get('subject', '')
                    predicate = item.get('predicate', '')
                    object_val = item.get('object', '')
                    print(f"   {i}. {subject} → [{predicate}] → {object_val}")
                else:
                    print(f"   {i}. {item}")
        else:
            print(f"\n📊 沒有檢索到相關知識")
        
        print(f"\n🤖 混合RAG生成的詳細回答:")
        print(f"{result['answer']}")
        
        print(f"\n⏱️ 執行時間: {result['total_time']:.2f} 秒")
    
    def _compare_modes(self, query: str):
        """
        比較三種模式的回答
        """
        print(f"\n{'='*70}")
        print(f"三模式比較：{query}")
        print(f"{'='*70}")
        
        # 1. 改進的LangChain模式
        print(f"\n【模式一：改進的LangChain】")
        print("-" * 40)
        result1 = self.answer_question(query, use_rag=False, use_langchain=True)
        print(f"回答: {result1['answer']}")
        print(f"時間: {result1['total_time']:.2f}s | 知識項目: {result1['knowledge_items_count']}")
        
        # 2. 混合RAG模式
        print(f"\n【模式二：混合RAG (推薦)】")
        print("-" * 40)
        result2 = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid=True)
        print(f"回答: {result2['answer']}")
        print(f"時間: {result2['total_time']:.2f}s | 知識項目: {result2['knowledge_items_count']}")
        
        # 3. 傳統RAG模式
        print(f"\n【模式三：傳統RAG】")
        print("-" * 40)
        result3 = self.answer_question(query, use_rag=True, use_langchain=False)
        print(f"回答: {result3['answer']}")
        print(f"時間: {result3['total_time']:.2f}s | 知識項目: {result3['knowledge_items_count']}")
        
        print(f"\n💡 推薦使用混合RAG模式獲得最佳效果")
    
    def _print_vector_result(self, result: Dict[str, Any]):
        """
        格式化打印純向量RAG結果
        """
        print(f"\n{'='*60}")
        print(f"問題: {result['query']}")
        print(f"【純向量RAG模式】")
        print(f"{'='*60}")
        
        if result['vector_results']:
            print(f"\n📊 從向量數據庫檢索到的知識 ({len(result['vector_results'])} 項):")
            for i, vr in enumerate(result['vector_results'], 1):
                print(f"   {i}. [相似度: {vr['similarity_score']:.3f}] {vr['content'][:100]}...")
                print(f"      來源: {vr['metadata'].get('source_file', 'Unknown')}")
        else:
            print(f"\n📊 沒有檢索到相關知識")
        
        print(f"\n🤖 向量RAG回答:")
        print(f"{result['answer']}")
        
        print(f"\n⏱️ 執行時間:")
        print(f"   檢索時間: {result['retrieval_time']:.2f} 秒")
        print(f"   生成時間: {result['generation_time']:.2f} 秒")
        print(f"   總時間: {result['total_time']:.2f} 秒")
    
    def _print_hybrid_all_result(self, result: Dict[str, Any]):
        """
        格式化打印全混合RAG結果
        """
        print(f"\n{'='*70}")
        print(f"問題: {result['query']}")
        print(f"【全混合RAG模式】知識圖譜 + 向量檢索")
        print(f"{'='*70}")
        
        if result['cypher_query']:
            print(f"\n🔧 知識圖譜Cypher查詢:")
            clean_query = result['cypher_query'].replace('cypher\n', '').strip()
            print(f"   {clean_query}")
        
        if result['context_data']:
            print(f"\n📊 知識圖譜檢索結果 ({len(result['context_data'])} 項):")
            for i, item in enumerate(result['context_data'], 1):
                if isinstance(item, dict):
                    subject = item.get('subject', '')
                    predicate = item.get('predicate', '')
                    object_val = item.get('object', '')
                    print(f"   {i}. {subject} → [{predicate}] → {object_val}")
                else:
                    print(f"   {i}. {item}")
        
        if result['vector_results']:
            print(f"\n🔍 向量檢索結果 ({len(result['vector_results'])} 項):")
            for i, vr in enumerate(result['vector_results'], 1):
                print(f"   {i}. [相似度: {vr['similarity_score']:.3f}] {vr['content'][:80]}...")
                print(f"      來源: {vr['metadata'].get('source_file', 'Unknown')}")
        
        print(f"\n🤖 全混合RAG回答:")
        print(f"{result['answer']}")
        
        print(f"\n⏱️ 執行時間: {result['total_time']:.2f} 秒")
        print(f"   (檢索: {result['retrieval_time']:.2f}s, 生成: {result['generation_time']:.2f}s)")
    
    def _compare_all_modes(self, query: str):
        """
        比較所有可用模式的回答
        """
        print(f"\n{'='*80}")
        print(f"多模式比較：{query}")
        print(f"{'='*80}")
        
        # 1. 改進的LangChain模式
        print(f"\n【模式一：知識圖譜LangChain】")
        print("-" * 40)
        result1 = self.answer_question(query, use_rag=False, use_langchain=True)
        print(f"回答: {result1['answer'][:200]}...")
        print(f"時間: {result1['total_time']:.2f}s | 知識項目: {result1['knowledge_items_count']}")
        
        # 2. 混合RAG模式
        print(f"\n【模式二：知識圖譜混合RAG】")
        print("-" * 40)
        result2 = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid=True)
        print(f"回答: {result2['answer'][:200]}...")
        print(f"時間: {result2['total_time']:.2f}s | 知識項目: {result2['knowledge_items_count']}")
        
        # 3. 純向量RAG模式
        if self.vector_available:
            print(f"\n【模式三：純向量RAG】")
            print("-" * 40)
            result3 = self.answer_question(query, use_rag=False, use_langchain=False, use_vector=True)
            print(f"回答: {result3['answer'][:200]}...")
            print(f"時間: {result3['total_time']:.2f}s | 知識項目: {result3['knowledge_items_count']}")
            
            # 4. 全混合模式
            print(f"\n【模式四：全混合RAG (推薦)】")
            print("-" * 40)
            result4 = self.answer_question(query, use_rag=False, use_langchain=False, use_hybrid_all=True)
            print(f"回答: {result4['answer'][:200]}...")
            print(f"時間: {result4['total_time']:.2f}s | 知識項目: {result4['knowledge_items_count']}")
            
            print(f"\n💡 推薦使用全混合RAG模式獲得最佳效果")
        else:
            print(f"\n⚠️  向量RAG不可用，僅比較知識圖譜模式")
    
    def close(self):
        """
        關閉連接
        """
        self.knowledge_retriever.close()
        if self.vector_rag_processor:
            self.vector_rag_processor.close() 