from neo4j import GraphDatabase
import csv
import os
import time

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def import_triples(self, triples_file):
        """從 CSV 文件導入三元組到 Neo4j"""
        with self.driver.session() as session:
            # 清空數據庫
            session.run("MATCH (n) DETACH DELETE n")
            print("已清空現有數據")
            
            # 讀取 CSV 文件
            with open(triples_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳過標題行
                
                count = 0
                for row in reader:
                    if len(row) >= 4:
                        subject, predicate, obj, source = row
                        
                        # 創建主體和客體節點，以及它們之間的關係
                        query = """
                        MERGE (s:Entity {name: $subject})
                        MERGE (o:Entity {name: $object})
                        CREATE (s)-[r:RELATION {name: $predicate, source: $source}]->(o)
                        RETURN s, r, o
                        """
                        
                        session.run(query, subject=subject, predicate=predicate, object=obj, source=source)
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"已導入 {count} 個三元組...")
            
            print(f"總共導入了 {count} 個三元組")
    
    def count_nodes_and_relationships(self):
        """計算節點和關係的數量"""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            return node_count, rel_count

def wait_for_neo4j(uri, user, password, max_retries=30, retry_interval=5):
    """等待 Neo4j 服務啟動"""
    print("等待 Neo4j 服務啟動...")
    retries = 0
    while retries < max_retries:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            driver.close()
            print("Neo4j 服務已啟動")
            return True
        except Exception as e:
            print(f"Neo4j 服務未就緒: {e}")
            retries += 1
            print(f"等待 {retry_interval} 秒後重試... ({retries}/{max_retries})")
            time.sleep(retry_interval)
    
    print("無法連接到 Neo4j 服務")
    return False

if __name__ == "__main__":
    # Neo4j 連接參數
    uri = "bolt://neo4j:7687"  # Docker 容器中的 Neo4j 實例
    user = "neo4j"
    password = "password123"
    
    # 三元組文件路徑
    triples_file = "/app/data/processed/triples.csv"
    
    # 等待 Neo4j 服務啟動
    if not wait_for_neo4j(uri, user, password):
        print("無法連接到 Neo4j，退出程序")
        exit(1)
    
    # 確認三元組文件存在
    if not os.path.exists(triples_file):
        print(f"三元組文件不存在: {triples_file}")
        print("請先運行 sentence_triplet_extractor.py 生成三元組")
        exit(1)
    
    # 導入三元組
    importer = Neo4jImporter(uri, user, password)
    try:
        importer.import_triples(triples_file)
        
        # 計算導入的節點和關係數量
        node_count, rel_count = importer.count_nodes_and_relationships()
        print(f"成功導入到 Neo4j: {node_count} 個節點和 {rel_count} 個關係")
    finally:
        importer.close() 