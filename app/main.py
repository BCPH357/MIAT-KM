print("Hello World")
import time
import os
from neo4j import GraphDatabase
from rag_system import RAGSystem

# 連接到 Neo4j
def connect_to_neo4j():
    uri = "bolt://neo4j:7687"
    user = "neo4j"
    password = "password123"
    
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

# 創建簡單的知識圖譜
def create_knowledge_graph(driver):
    with driver.session() as session:
        # 清除現有數據
        session.run("MATCH (n) DETACH DELETE n")
        print("已清除現有數據")
        
        # 創建一些節點和關係
        session.run("""
        CREATE (ai:Topic {name: '人工智能'})
        CREATE (ml:Topic {name: '機器學習'})
        CREATE (dl:Topic {name: '深度學習'})
        CREATE (nlp:Topic {name: '自然語言處理'})
        CREATE (kg:Topic {name: '知識圖譜'})
        
        CREATE (ai)-[:INCLUDES]->(ml)
        CREATE (ml)-[:INCLUDES]->(dl)
        CREATE (dl)-[:USED_IN]->(nlp)
        CREATE (nlp)-[:USES]->(kg)
        """)
        
        print("知識圖譜已創建!")

# 查詢知識圖譜
def query_knowledge_graph(driver):
    print("\n執行查詢測試:")
    
    with driver.session() as session:
        # 查詢所有主題
        print("\n1. 查詢所有主題:")
        result = session.run("MATCH (t:Topic) RETURN t.name AS name")
        for record in result:
            print(f"  - {record['name']}")
        
        # 查詢關係
        print("\n2. 查詢 '人工智能' 包含的主題:")
        result = session.run("""
        MATCH (ai:Topic {name: '人工智能'})-[:INCLUDES]->(t:Topic)
        RETURN t.name AS name
        """)
        for record in result:
            print(f"  - {record['name']}")
        
        # 查詢路徑
        print("\n3. 查詢從 '人工智能' 到 '知識圖譜' 的路徑:")
        result = session.run("""
        MATCH path = (ai:Topic {name: '人工智能'})-[*]->(kg:Topic {name: '知識圖譜'})
        RETURN [node in nodes(path) | node.name] AS path
        """)
        for record in result:
            print(f"  - 路徑: {' -> '.join(record['path'])}")

# 檢查 PDF 目錄
def check_pdf_directory():
    pdf_dir = "/app/data/pdf"
    if not os.path.exists(pdf_dir):
        print(f"PDF 目錄不存在: {pdf_dir}")
        return False
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"PDF 目錄中沒有 PDF 文件，請先上傳 PDF 文件到 {pdf_dir}")
        return False
    
    print(f"找到 {len(pdf_files)} 個 PDF 文件")
    return True

# 顯示菜單
def show_menu():
    print("\n=== 知識圖譜應用菜單 ===")
    print("1. 創建示例知識圖譜")
    print("2. 從 PDF 文件提取三元組")
    print("3. 將三元組導入到 Neo4j")
    print("4. 查詢知識圖譜")
    print("5. RAG 問答系統")
    print("6. 退出")
    choice = input("請選擇操作 (1-6): ")
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
                # 創建知識圖譜
                create_knowledge_graph(neo4j_driver)
                
            elif choice == '2':
                # 檢查 PDF 目錄
                if check_pdf_directory():
                    # 從 PDF 提取三元組
                    print("開始從 PDF 提取三元組...")
                    print("使用基於句子的 DeepSeek 抽取器...")
                    os.system("python /app/sentence_triplet_extractor.py")
                
            elif choice == '3':
                # 將三元組導入到 Neo4j
                print("開始將三元組導入到 Neo4j...")
                os.system("python /app/import_to_neo4j.py")
                
            elif choice == '4':
                # 查詢知識圖譜
                query_knowledge_graph(neo4j_driver)
                
            elif choice == '5':
                # RAG 問答系統
                print("啟動 RAG 問答系統...")
                try:
                    rag_system = RAGSystem()
                    rag_system.interactive_qa()
                    rag_system.close()
                except Exception as e:
                    print(f"RAG 系統啟動失敗: {e}")
                
            elif choice == '6':
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