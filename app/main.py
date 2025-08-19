import time
import os
from neo4j import GraphDatabase
from rag_system import RAGSystem
from vector_rag_processor import VectorRAGProcessor
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, PDF_DIR, MARKDOWN_DIR, OLLAMA_MODEL

# 連接到 Neo4j
def connect_to_neo4j():
    uri = NEO4J_URI
    user = NEO4J_USER
    password = NEO4J_PASSWORD
    
    print("正在連接到 Neo4j...")
    
    # 嘗試連接直到成功
    while True:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            # 測試連接
            with driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            print("成功連接到 Neo4j!")
            return driver
        except Exception as e:
            print(f"無法連接到 Neo4j: {e}")
            print("5秒後重試...")
            time.sleep(5)



# 檢查文件目錄
def check_files_directory():
    pdf_exists = os.path.exists(PDF_DIR)
    md_exists = os.path.exists(MARKDOWN_DIR)
    
    if not pdf_exists and not md_exists:
        print(f"文件目錄不存在: {PDF_DIR} 和 {MARKDOWN_DIR}")
        return False
    
    total_files = 0
    
    if pdf_exists:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
        total_files += len(pdf_files)
        if pdf_files:
            print(f"找到 {len(pdf_files)} 個 PDF 文件")
    
    if md_exists:
        md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
        total_files += len(md_files)
        if md_files:
            print(f"找到 {len(md_files)} 個 Markdown 文件")
    
    if total_files == 0:
        print(f"沒有找到任何支持的文件，請上傳 PDF 或 Markdown 文件")
        return False
    
    print(f"總共找到 {total_files} 個支持的文件")
    return True

# 檢查 PDF 目錄 (向後兼容)
def check_pdf_directory():
    return check_files_directory()

# 顯示菜單
def show_menu():
    print("\n=== 知識圖譜應用菜單 ===")
    print("1. 從文件提取三元組 (PDF 和 Markdown)")
    print("2. 將三元組導入到 Neo4j")
    print("3. RAG 問答系統")
    print("4. Vector RAG 文檔預處理")
    print("5. 退出")
    choice = input("請選擇操作 (1-5): ")
    return choice

# 主函數
def main():
    print("啟動 Neo4j 知識圖譜應用...")
    
    # 連接 Neo4j
    neo4j_driver = connect_to_neo4j()
    
    try:
        while True:
            choice = show_menu()
            
            if choice == '1':
                # 檢查文件目錄
                if check_files_directory():
                    # 從文件提取三元組
                    print("開始從文件提取三元組...")
                    print("支持 PDF 和 Markdown 文件")
                    print(f"使用 {OLLAMA_MODEL} 模型進行抽取...")
                    os.system("python /app/sentence_triplet_extractor.py")
                
            elif choice == '2':
                # 將三元組導入到 Neo4j
                print("開始將三元組導入到 Neo4j...")
                os.system("python /app/import_to_neo4j.py")
                
            elif choice == '3':
                # RAG 問答系統
                print("啟動 RAG 問答系統...")
                try:
                    rag_system = RAGSystem()
                    rag_system.interactive_qa()
                    rag_system.close()
                except Exception as e:
                    print(f"RAG 系統啟動失敗: {e}")
                
            elif choice == '4':
                # Vector RAG 文檔預處理
                print("啟動 Vector RAG 文檔預處理...")
                handle_vector_rag_preprocessing()
                
            elif choice == '5':
                # 退出
                print("應用關閉中...")
                break
                
            else:
                print("無效的選擇，請重新輸入")
            
            if choice not in ['4']:  # Vector RAG預處理有自己的循環，不需要暫停
                input("\n按 Enter 鍵繼續...")
            
    except KeyboardInterrupt:
        print("\n接收到中斷信號，應用關閉中...")
    except Exception as e:
        print(f"\n發生錯誤: {e}")
    finally:
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j 連接已關閉")

def handle_vector_rag_preprocessing():
    """
    處理 Vector RAG 文檔預處理功能
    """
    try:
        # 初始化 Vector RAG 處理器
        print("初始化 Vector RAG 處理器...")
        processor = VectorRAGProcessor()
        
        while True:
            print("\n=== Vector RAG 文檔預處理 ===")
            print("1. 處理所有文檔 (清空現有數據)")
            print("2. 處理所有文檔 (增量模式)")
            print("3. 查看數據庫統計")
            print("4. 清空向量數據庫")
            print("5. 測試向量搜索")
            print("6. 返回主菜單")
            
            sub_choice = input("請選擇操作 (1-6): ")
            
            if sub_choice == '1':
                # 處理所有文檔 (清空現有)
                if check_files_directory():
                    print("開始處理文檔（清空模式）...")
                    result = processor.process_documents_from_directories(clear_existing=True)
                    print_processing_result(result)
                    
            elif sub_choice == '2':
                # 處理所有文檔 (增量模式)
                if check_files_directory():
                    print("開始處理文檔（增量模式）...")
                    result = processor.process_documents_from_directories(clear_existing=False)
                    print_processing_result(result)
                    
            elif sub_choice == '3':
                # 查看數據庫統計
                print("獲取數據庫統計信息...")
                stats = processor.get_database_stats()
                print_database_stats(stats)
                
            elif sub_choice == '4':
                # 清空向量數據庫
                confirm = input("確定要清空向量數據庫嗎？(y/N): ")
                if confirm.lower() == 'y':
                    success = processor.clear_database()
                    if success:
                        print("向量數據庫清空成功")
                    else:
                        print("向量數據庫清空失敗")
                else:
                    print("取消操作")
                    
            elif sub_choice == '5':
                # 測試向量搜索
                test_query = input("請輸入測試查詢: ")
                if test_query.strip():
                    print("執行向量搜索...")
                    search_results = processor.search_documents(test_query, n_results=3)
                    print_search_results(search_results)
                    
            elif sub_choice == '6':
                # 返回主菜單
                break
                
            else:
                print("無效的選擇，請重新輸入")
        
        # 關閉處理器
        processor.close()
        
    except Exception as e:
        print(f"Vector RAG 處理器出現錯誤: {e}")
        import traceback
        traceback.print_exc()

def print_processing_result(result):
    """
    打印處理結果
    """
    print(f"\n=== 處理結果 ===")
    print(f"總文件數: {result['total_files']}")
    print(f"成功處理: {result['processed_files']}")
    print(f"總chunks: {result['total_chunks']}")
    print(f"成功存儲: {result['successful_chunks']}")
    print(f"處理時間: {result['processing_time']:.2f}秒")
    print(f"Embedding時間: {result['embedding_time']:.2f}秒")
    print(f"存儲時間: {result['storage_time']:.2f}秒")
    
    if result['failed_files']:
        print(f"\n失敗文件 ({len(result['failed_files'])}個):")
        for failed in result['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")

def print_database_stats(stats):
    """
    打印數據庫統計信息
    """
    print(f"\n=== 數據庫統計 ===")
    print(f"集合名稱: {stats['collection_name']}")
    print(f"總chunks數: {stats['total_chunks']}")
    print(f"獨特文件數: {stats['unique_files']}")
    print(f"數據庫路徑: {stats['database_path']}")
    
    if stats['file_types']:
        print(f"\n文件類型統計:")
        for file_type, count in stats['file_types'].items():
            print(f"  {file_type}: {count} chunks")
    
    if 'error' in stats:
        print(f"\n錯誤: {stats['error']}")

def print_search_results(search_results):
    """
    打印搜索結果
    """
    print(f"\n=== 搜索結果 ===")
    print(f"查詢: {search_results['query']}")
    print(f"找到 {search_results['total_results']} 個相關文檔\n")
    
    for result in search_results['results']:
        print(f"排名 {result['rank']}:")
        print(f"  相似度: {result['similarity_score']:.3f}")
        print(f"  來源: {result['metadata'].get('source_file', 'Unknown')}")
        print(f"  內容預覽: {result['content'][:500]}...")
        print("-" * 50)
    
    if 'error' in search_results:
        print(f"\n搜索錯誤: {search_results['error']}")

if __name__ == "__main__":
    main() 