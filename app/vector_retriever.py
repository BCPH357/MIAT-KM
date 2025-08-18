import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    ChromaDB向量檢索器
    封裝ChromaDB操作，提供文檔向量存儲和檢索功能
    """
    
    def __init__(self, 
                 db_path: str = CHROMA_DB_PATH,
                 collection_name: str = CHROMA_COLLECTION_NAME):
        """
        初始化向量檢索器
        
        Args:
            db_path: ChromaDB數據庫路徑
            collection_name: 集合名稱
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        logger.info(f"初始化向量檢索器 - 數據庫路徑: {db_path}")
        self._init_chroma_client()
    
    def _init_chroma_client(self):
        """初始化ChromaDB客戶端"""
        try:
            # 確保數據庫目錄存在
            os.makedirs(self.db_path, exist_ok=True)
            
            # 初始化ChromaDB客戶端
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 獲取或創建集合
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"成功連接到現有集合: {self.collection_name}")
            except ValueError:
                # 集合不存在，創建新集合
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "MIAT Knowledge Management System documents"}
                )
                logger.info(f"創建新集合: {self.collection_name}")
            
            logger.info("ChromaDB客戶端初始化成功")
            
        except Exception as e:
            logger.error(f"ChromaDB客戶端初始化失敗: {e}")
            raise RuntimeError(f"無法初始化ChromaDB: {e}")
    
    def add_documents(self, 
                     documents: List[str],
                     embeddings: List[List[float]],
                     metadatas: List[Dict] = None,
                     ids: List[str] = None) -> bool:
        """
        添加文檔到向量數據庫
        
        Args:
            documents: 文檔文本列表
            embeddings: 對應的向量embeddings列表
            metadatas: 元數據列表
            ids: 文檔ID列表
            
        Returns:
            是否成功添加
        """
        try:
            if not documents or not embeddings:
                logger.warning("文檔或embeddings為空，跳過添加")
                return False
            
            if len(documents) != len(embeddings):
                logger.error("文檔數量與embeddings數量不匹配")
                return False
            
            # 生成ID（如果未提供）
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 生成元數據（如果未提供）
            if metadatas is None:
                metadatas = [{"index": i} for i in range(len(documents))]
            
            # 添加到ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"成功添加 {len(documents)} 個文檔到向量數據庫")
            return True
            
        except Exception as e:
            logger.error(f"添加文檔到向量數據庫失敗: {e}")
            return False
    
    def search_similar(self, 
                      query_embedding: List[float],
                      n_results: int = 5,
                      where: Dict = None,
                      include: List[str] = None) -> Dict[str, Any]:
        """
        搜索相似文檔
        
        Args:
            query_embedding: 查詢向量
            n_results: 返回結果數量
            where: 元數據過濾條件
            include: 包含的字段
            
        Returns:
            搜索結果字典
        """
        try:
            if include is None:
                include = ["documents", "metadatas", "distances"]
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include
            )
            
            # 格式化結果
            formatted_results = {
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'distances': results.get('distances', [[]])[0],
                'ids': results.get('ids', [[]])[0]
            }
            
            logger.info(f"檢索到 {len(formatted_results['documents'])} 個相似文檔")
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量檢索失敗: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
    
    def search_by_text_with_embedder(self, 
                                   query_text: str,
                                   embedder,
                                   n_results: int = 5,
                                   where: Dict = None) -> Dict[str, Any]:
        """
        通過文本搜索（需要embedder）
        
        Args:
            query_text: 查詢文本
            embedder: embedding模型實例
            n_results: 返回結果數量
            where: 元數據過濾條件
            
        Returns:
            搜索結果字典
        """
        try:
            # 將查詢文本轉換為向量
            query_embedding = embedder.encode(query_text)
            
            # 搜索相似文檔
            return self.search_similar(
                query_embedding=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                n_results=n_results,
                where=where
            )
            
        except Exception as e:
            logger.error(f"文本檢索失敗: {e}")
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        刪除指定ID的文檔
        
        Args:
            ids: 要刪除的文檔ID列表
            
        Returns:
            是否成功刪除
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"成功刪除 {len(ids)} 個文檔")
            return True
            
        except Exception as e:
            logger.error(f"刪除文檔失敗: {e}")
            return False
    
    def update_documents(self, 
                        ids: List[str],
                        documents: List[str] = None,
                        embeddings: List[List[float]] = None,
                        metadatas: List[Dict] = None) -> bool:
        """
        更新文檔
        
        Args:
            ids: 文檔ID列表
            documents: 新文檔內容（可選）
            embeddings: 新embeddings（可選）
            metadatas: 新元數據（可選）
            
        Returns:
            是否成功更新
        """
        try:
            update_params = {"ids": ids}
            
            if documents is not None:
                update_params["documents"] = documents
            if embeddings is not None:
                update_params["embeddings"] = embeddings
            if metadatas is not None:
                update_params["metadatas"] = metadatas
            
            self.collection.update(**update_params)
            logger.info(f"成功更新 {len(ids)} 個文檔")
            return True
            
        except Exception as e:
            logger.error(f"更新文檔失敗: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        獲取集合信息
        
        Returns:
            集合信息字典
        """
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'metadata': self.collection.metadata
            }
        except Exception as e:
            logger.error(f"獲取集合信息失敗: {e}")
            return {'name': self.collection_name, 'count': 0, 'metadata': {}}
    
    def clear_collection(self) -> bool:
        """
        清空集合中的所有文檔
        
        Returns:
            是否成功清空
        """
        try:
            # 獲取所有文檔ID
            results = self.collection.get()
            if results and 'ids' in results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"成功清空集合 {self.collection_name}")
            else:
                logger.info(f"集合 {self.collection_name} 已經為空")
            return True
            
        except Exception as e:
            logger.error(f"清空集合失敗: {e}")
            return False
    
    def reset_database(self) -> bool:
        """
        重置整個數據庫
        
        Returns:
            是否成功重置
        """
        try:
            self.client.reset()
            logger.info("數據庫重置成功")
            
            # 重新初始化
            self._init_chroma_client()
            return True
            
        except Exception as e:
            logger.error(f"數據庫重置失敗: {e}")
            return False
    
    def batch_search(self, 
                    query_embeddings: List[List[float]],
                    n_results: int = 5) -> List[Dict[str, Any]]:
        """
        批量搜索
        
        Args:
            query_embeddings: 查詢向量列表
            n_results: 每個查詢返回的結果數量
            
        Returns:
            搜索結果列表
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化結果
            formatted_results = []
            for i in range(len(query_embeddings)):
                formatted_result = {
                    'documents': results['documents'][i] if i < len(results['documents']) else [],
                    'metadatas': results['metadatas'][i] if i < len(results['metadatas']) else [],
                    'distances': results['distances'][i] if i < len(results['distances']) else [],
                    'ids': results['ids'][i] if i < len(results['ids']) else []
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"批量檢索完成: {len(query_embeddings)} 個查詢")
            return formatted_results
            
        except Exception as e:
            logger.error(f"批量檢索失敗: {e}")
            return []
    
    def get_documents_by_metadata(self, where: Dict) -> Dict[str, Any]:
        """
        根據元數據獲取文檔
        
        Args:
            where: 元數據過濾條件
            
        Returns:
            匹配的文檔
        """
        try:
            results = self.collection.get(where=where)
            return results
            
        except Exception as e:
            logger.error(f"根據元數據獲取文檔失敗: {e}")
            return {'documents': [], 'metadatas': [], 'ids': []}
    
    def close(self):
        """關閉連接和清理資源"""
        try:
            if self.client:
                # ChromaDB客戶端會自動處理持久化
                logger.info("ChromaDB連接已關閉")
        except Exception as e:
            logger.error(f"關閉ChromaDB連接時出錯: {e}")
    
    def __del__(self):
        """析構函數"""
        self.close()