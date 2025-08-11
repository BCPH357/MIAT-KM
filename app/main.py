print("Hello World")
import time
import os
from neo4j import GraphDatabase
from rag_system import RAGSystem
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
    print("4. 退出")
    choice = input("請選擇操作 (1-4): ")
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
                # 退出
                print("應用關閉中...")
                break
                
            else:
                print("無效的選擇，請重新輸入")
            
            input("\n按 Enter 鍵繼續...")
            
    except KeyboardInterrupt:
        print("\n接收到中斷信號，應用關閉中...")
    except Exception as e:
        print(f"\n發生錯誤: {e}")
    finally:
        if neo4j_driver:
            neo4j_driver.close()
            print("Neo4j 連接已關閉")

if __name__ == "__main__":
    main() 