#!/usr/bin/env python3
"""
RAG 系統調試腳本
用於查看查詢如何轉換為 Neo4j 查詢語句
"""

from knowledge_retriever import Neo4jKnowledgeRetriever
from rag_system import RAGSystem
import sys

def debug_query_transformation(query: str):
    """調試查詢轉換過程"""
    print(f"=== 調試查詢: {query} ===")
    
    # 1. 初始化知識檢索器
    retriever = Neo4jKnowledgeRetriever(
        uri="bolt://neo4j:7687",
        user="neo4j", 
        password="password123"
    )
    
    try:
        # 2. 提取關鍵字
        keywords = retriever._extract_keywords(query)
        print(f"\n1. 提取的關鍵字: {keywords}")
        
        if not keywords:
            print("❌ 沒有提取到關鍵字！")
            return
        
        # 3. 顯示生成的 Cypher 查詢
        cypher_query = """
        MATCH (s:Entity)-[r:RELATION]->(o:Entity)
        WHERE ANY(keyword IN $keywords WHERE 
            s.name CONTAINS keyword OR 
            o.name CONTAINS keyword OR 
            r.name CONTAINS keyword
        )
        RETURN s.name as subject, r.name as predicate, o.name as object, r.source as source
        LIMIT $limit
        """
        
        print(f"\n2. 生成的 Cypher 查詢:")
        print(cypher_query)
        print(f"   參數: keywords={keywords}, limit=10")
        
        # 4. 執行查詢並顯示結果
        knowledge_items = retriever.search_related_knowledge(query, limit=10)
        print(f"\n3. 查詢結果數量: {len(knowledge_items)}")
        
        if knowledge_items:
            print("\n4. 查詢結果詳情:")
            for i, item in enumerate(knowledge_items, 1):
                print(f"   {i}. {item['subject']} --[{item['predicate']}]--> {item['object']}")
                print(f"      來源: {item['source']}")
        else:
            print("\n❌ 查詢結果為空！")
            
            # 5. 檢查數據庫中是否有相關數據
            print("\n5. 檢查數據庫狀態:")
            with retriever.driver.session() as session:
                # 檢查總節點數
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                print(f"   總節點數: {node_count}")
                
                # 檢查總關係數
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                print(f"   總關係數: {rel_count}")
                
                # 檢查是否有 Entity 節點
                entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                print(f"   Entity 節點數: {entity_count}")
                
                # 檢查是否有 RELATION 關係
                relation_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()["count"]
                print(f"   RELATION 關係數: {relation_count}")
                
                # 顯示一些示例數據
                print(f"\n6. 示例數據 (前5個):")
                sample_data = session.run("""
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                RETURN s.name as subject, r.name as predicate, o.name as object
                LIMIT 5
                """)
                
                for i, record in enumerate(sample_data, 1):
                    print(f"   {i}. {record['subject']} --[{record['predicate']}]--> {record['object']}")
                
                # 7. 檢查是否有包含關鍵字的數據
                print(f"\n7. 檢查包含關鍵字的數據:")
                for keyword in keywords:
                    keyword_query = """
                    MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                    WHERE s.name CONTAINS $keyword OR o.name CONTAINS $keyword OR r.name CONTAINS $keyword
                    RETURN count(*) as count
                    """
                    count = session.run(keyword_query, keyword=keyword).single()["count"]
                    print(f"   關鍵字 '{keyword}': {count} 個匹配")
        
        # 8. 測試綜合搜索
        print(f"\n8. 綜合搜索結果:")
        comprehensive_result = retriever.comprehensive_search(query)
        print(comprehensive_result)
        
    except Exception as e:
        print(f"❌ 調試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        retriever.close()

def main():
    """主函數"""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("請輸入要調試的查詢: ")
    
    debug_query_transformation(query)

if __name__ == "__main__":
    main() 