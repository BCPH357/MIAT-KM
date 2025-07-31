from neo4j import GraphDatabase
import re
from typing import List, Dict, Tuple

class Neo4jKnowledgeRetriever:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def search_related_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """
        根據用戶查詢搜索相關的知識三元組
        """
        # 提取查詢中的關鍵字
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        with self.driver.session() as session:
            # 搜索包含關鍵字的實體和關係
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
            
            result = session.run(cypher_query, keywords=keywords, limit=limit)
            
            knowledge_items = []
            for record in result:
                knowledge_items.append({
                    'subject': record['subject'],
                    'predicate': record['predicate'],
                    'object': record['object'],
                    'source': record['source']
                })
            
            return knowledge_items
    
    def search_by_entity(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """
        根據實體名稱搜索相關的所有關係
        """
        with self.driver.session() as session:
            cypher_query = """
            MATCH (e:Entity {name: $entity_name})-[r:RELATION]-(other:Entity)
            RETURN e.name as entity, r.name as relation, other.name as related_entity, 
                   r.source as source, 
                   CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END as direction
            LIMIT $limit
            """
            
            result = session.run(cypher_query, entity_name=entity_name, limit=limit)
            
            relations = []
            for record in result:
                relations.append({
                    'entity': record['entity'],
                    'relation': record['relation'],
                    'related_entity': record['related_entity'],
                    'source': record['source'],
                    'direction': record['direction']
                })
            
            return relations
    
    def search_paths(self, start_entity: str, end_entity: str, max_depth: int = 3) -> List[Dict]:
        """
        搜索兩個實體之間的路徑
        """
        with self.driver.session() as session:
            cypher_query = """
            MATCH path = (start:Entity {name: $start_entity})-[*1..$max_depth]-(end:Entity {name: $end_entity})
            RETURN [node in nodes(path) | node.name] as entities,
                   [rel in relationships(path) | rel.name] as relations
            LIMIT 5
            """
            
            result = session.run(cypher_query, 
                               start_entity=start_entity, 
                               end_entity=end_entity, 
                               max_depth=max_depth)
            
            paths = []
            for record in result:
                paths.append({
                    'entities': record['entities'],
                    'relations': record['relations']
                })
            
            return paths
    
    def get_entity_neighbors(self, entity_name: str, limit: int = 5) -> List[Dict]:
        """
        獲取實體的鄰居節點
        """
        with self.driver.session() as session:
            cypher_query = """
            MATCH (e:Entity {name: $entity_name})-[r:RELATION]-(neighbor:Entity)
            RETURN DISTINCT neighbor.name as neighbor, r.name as relation, r.source as source
            LIMIT $limit
            """
            
            result = session.run(cypher_query, entity_name=entity_name, limit=limit)
            
            neighbors = []
            for record in result:
                neighbors.append({
                    'neighbor': record['neighbor'],
                    'relation': record['relation'],
                    'source': record['source']
                })
            
            return neighbors
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        從查詢中提取關鍵字
        """
        # 移除標點符號並分割成詞
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        words = clean_query.split()
        
        # 過濾掉過短的詞和常見停用詞
        stopwords = {'的', '是', '在', '和', '與', '或', '但', '如果', '因為', '所以', '什麼', '怎麼', '為什麼', '哪個', '這個', '那個'}
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        
        return keywords
    
    def format_knowledge_for_llm(self, knowledge_items: List[Dict]) -> str:
        """
        將檢索到的知識格式化為適合 LLM 的文本
        """
        if not knowledge_items:
            return "沒有找到相關的知識。"
        
        formatted_text = "以下是從知識庫檢索到的相關信息：\n\n"
        
        for i, item in enumerate(knowledge_items, 1):
            formatted_text += f"{i}. {item['subject']} {item['predicate']} {item['object']}"
            if item.get('source'):
                formatted_text += f" (來源: {item['source']})"
            formatted_text += "\n"
        
        return formatted_text
    
    def comprehensive_search(self, query: str) -> str:
        """
        綜合搜索：結合關鍵字搜索和實體搜索
        """
        # 首先進行關鍵字搜索
        knowledge_items = self.search_related_knowledge(query, limit=8)
        
        # 如果關鍵字搜索結果較少，嘗試提取可能的實體名稱進行實體搜索
        if len(knowledge_items) < 3:
            keywords = self._extract_keywords(query)
            for keyword in keywords[:2]:  # 只檢查前兩個關鍵字作為實體
                entity_relations = self.search_by_entity(keyword, limit=3)
                # 將實體關係轉換為知識項目格式
                for rel in entity_relations:
                    if rel['direction'] == 'outgoing':
                        knowledge_items.append({
                            'subject': rel['entity'],
                            'predicate': rel['relation'],
                            'object': rel['related_entity'],
                            'source': rel['source']
                        })
                    else:
                        knowledge_items.append({
                            'subject': rel['related_entity'],
                            'predicate': rel['relation'],
                            'object': rel['entity'],
                            'source': rel['source']
                        })
        
        # 去重並限制結果數量
        unique_items = []
        seen = set()
        for item in knowledge_items:
            key = f"{item['subject']}-{item['predicate']}-{item['object']}"
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
                if len(unique_items) >= 10:
                    break
        
        return self.format_knowledge_for_llm(unique_items) 