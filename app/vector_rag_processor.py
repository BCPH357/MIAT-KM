import os
import time
from typing import List, Dict, Any, Optional
import logging
from vector_embedder import QwenEmbedder
from document_chunker import DocumentChunker
from vector_retriever import VectorRetriever
from config import PDF_DIR, MARKDOWN_DIR, CHROMA_DB_PATH
import uuid

logger = logging.getLogger(__name__)

class VectorRAGProcessor:
    """
    Vector RAG處理器
    整合文檔分塊、向量化和存儲的完整流程
    """
    
    def __init__(self):
        """初始化Vector RAG處理器"""
        logger.info("初始化Vector RAG處理器...")
        
        try:
            # 初始化各個組件
            self.embedder = QwenEmbedder()
            self.chunker = DocumentChunker()
            self.retriever = VectorRetriever()
            
            logger.info("Vector RAG處理器初始化成功")
            
        except Exception as e:
            logger.error(f"Vector RAG處理器初始化失敗: {e}")
            raise RuntimeError(f"無法初始化Vector RAG處理器: {e}")
    
    def process_documents_from_directories(self, 
                                         pdf_dir: str = PDF_DIR,
                                         markdown_dir: str = MARKDOWN_DIR,
                                         clear_existing: bool = False) -> Dict[str, Any]:
        """
        處理指定目錄中的所有文檔
        
        Args:
            pdf_dir: PDF文件目錄
            markdown_dir: Markdown文件目錄
            clear_existing: 是否清空現有數據
            
        Returns:
            處理結果統計
        """
        start_time = time.time()
        
        result = {
            'total_files': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_files': [],
            'processing_time': 0,
            'embedding_time': 0,
            'storage_time': 0
        }
        
        logger.info("開始批量處理文檔...")
        
        # 清空現有數據（如果需要）
        if clear_existing:
            logger.info("清空現有向量數據...")
            self.retriever.clear_collection()
        
        # 收集所有文件
        all_files = []
        
        # 掃描PDF文件
        if os.path.exists(pdf_dir):
            for file_name in os.listdir(pdf_dir):
                if file_name.lower().endswith('.pdf'):
                    all_files.append(os.path.join(pdf_dir, file_name))
        
        # 掃描Markdown文件
        if os.path.exists(markdown_dir):
            for file_name in os.listdir(markdown_dir):
                if file_name.lower().endswith(('.md', '.markdown')):
                    all_files.append(os.path.join(markdown_dir, file_name))
        
        result['total_files'] = len(all_files)
        logger.info(f"找到 {len(all_files)} 個文件待處理")
        
        if not all_files:
            logger.warning("沒有找到任何支持的文件")
            return result
        
        # 處理每個文件
        for file_path in all_files:
            try:
                file_result = self.process_single_file(file_path)
                
                if file_result['success']:
                    result['processed_files'] += 1
                    result['total_chunks'] += file_result['total_chunks']
                    result['successful_chunks'] += file_result['successful_chunks']
                    result['embedding_time'] += file_result['embedding_time']
                    result['storage_time'] += file_result['storage_time']
                else:
                    result['failed_files'].append({
                        'file': os.path.basename(file_path),
                        'error': file_result['error']
                    })
                
                logger.info(f"文件處理進度: {result['processed_files']}/{len(all_files)}")
                
            except Exception as e:
                logger.error(f"處理文件失敗 {file_path}: {e}")
                result['failed_files'].append({
                    'file': os.path.basename(file_path),
                    'error': str(e)
                })
        
        result['processing_time'] = time.time() - start_time
        
        logger.info(f"批量處理完成:")
        logger.info(f"  - 總文件數: {result['total_files']}")
        logger.info(f"  - 成功處理: {result['processed_files']}")
        logger.info(f"  - 總chunks: {result['total_chunks']}")
        logger.info(f"  - 成功存儲: {result['successful_chunks']}")
        logger.info(f"  - 處理時間: {result['processing_time']:.2f}秒")
        
        return result
    
    def process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        處理單個文件
        
        Args:
            file_path: 文件路徑
            
        Returns:
            處理結果
        """
        start_time = time.time()
        
        result = {
            'success': False,
            'file_path': file_path,
            'total_chunks': 0,
            'successful_chunks': 0,
            'embedding_time': 0,
            'storage_time': 0,
            'error': None
        }
        
        try:
            file_name = os.path.basename(file_path)
            logger.info(f"處理文件: {file_name}")
            
            # 步驟1: 文檔分塊
            chunks = self.chunker.chunk_file(file_path)
            result['total_chunks'] = len(chunks)
            
            if not chunks:
                result['error'] = "文檔分塊失敗或無有效內容"
                return result
            
            logger.info(f"文檔分塊完成: {len(chunks)} 個chunks")
            
            # 步驟2: 生成embeddings
            embedding_start = time.time()
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode(texts)
            result['embedding_time'] = time.time() - embedding_start
            
            logger.info(f"Embedding生成完成: {len(embeddings)} 個向量")
            
            # 步驟3: 準備存儲數據
            storage_start = time.time()
            
            # 生成唯一ID
            ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{chunk['chunk_id']}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                
                # 增強元數據
                metadata = chunk['metadata'].copy()
                metadata.update({
                    'chunk_index': i,
                    'chunk_size': chunk['chunk_size'],
                    'file_name': file_name,
                    'processed_time': time.time()
                })
                metadatas.append(metadata)
            
            # 步驟4: 存儲到向量數據庫
            success = self.retriever.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            result['storage_time'] = time.time() - storage_start
            
            if success:
                result['successful_chunks'] = len(chunks)
                result['success'] = True
                logger.info(f"文件處理成功: {file_name}")
            else:
                result['error'] = "向量存儲失敗"
                logger.error(f"文件處理失敗: {file_name} - 向量存儲失敗")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"處理文件時發生錯誤 {file_path}: {e}")
        
        return result
    
    def search_documents(self, 
                        query: str,
                        n_results: int = 5,
                        filter_metadata: Dict = None) -> Dict[str, Any]:
        """
        搜索相關文檔
        
        Args:
            query: 查詢文本
            n_results: 返回結果數量
            filter_metadata: 元數據過濾條件
            
        Returns:
            搜索結果
        """
        try:
            logger.info(f"執行向量搜索: {query}")
            
            # 使用嵌入器進行搜索
            results = self.retriever.search_by_text_with_embedder(
                query_text=query,
                embedder=self.embedder,
                n_results=n_results,
                where=filter_metadata
            )
            
            # 格式化結果
            formatted_results = []
            
            for i, (doc, metadata, distance, doc_id) in enumerate(zip(
                results['documents'],
                results['metadatas'],
                results['distances'],
                results['ids']
            )):
                formatted_result = {
                    'rank': i + 1,
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,  # 轉換為相似度分數
                    'distance': distance,
                    'document_id': doc_id
                }
                formatted_results.append(formatted_result)
            
            search_result = {
                'query': query,
                'total_results': len(formatted_results),
                'results': formatted_results
            }
            
            logger.info(f"向量搜索完成: 找到 {len(formatted_results)} 個相關文檔")
            return search_result
            
        except Exception as e:
            logger.error(f"向量搜索失敗: {e}")
            return {
                'query': query,
                'total_results': 0,
                'results': [],
                'error': str(e)
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        獲取數據庫統計信息
        
        Returns:
            數據庫統計信息
        """
        try:
            collection_info = self.retriever.get_collection_info()
            
            # 獲取文件類型統計
            all_docs = self.retriever.collection.get()
            file_types = {}
            source_files = set()
            
            if 'metadatas' in all_docs and all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    
                    source_file = metadata.get('source_file')
                    if source_file:
                        source_files.add(source_file)
            
            stats = {
                'collection_name': collection_info['name'],
                'total_chunks': collection_info['count'],
                'unique_files': len(source_files),
                'file_types': file_types,
                'database_path': self.retriever.db_path
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"獲取數據庫統計失敗: {e}")
            return {
                'collection_name': 'unknown',
                'total_chunks': 0,
                'unique_files': 0,
                'file_types': {},
                'database_path': self.retriever.db_path,
                'error': str(e)
            }
    
    def clear_database(self) -> bool:
        """
        清空向量數據庫
        
        Returns:
            是否成功清空
        """
        try:
            logger.info("清空向量數據庫...")
            success = self.retriever.clear_collection()
            if success:
                logger.info("向量數據庫清空成功")
            return success
            
        except Exception as e:
            logger.error(f"清空向量數據庫失敗: {e}")
            return False
    
    def close(self):
        """關閉所有連接和資源"""
        try:
            if hasattr(self, 'retriever'):
                self.retriever.close()
            
            if hasattr(self, 'embedder'):
                del self.embedder
            
            logger.info("Vector RAG處理器已關閉")
            
        except Exception as e:
            logger.error(f"關閉Vector RAG處理器時出錯: {e}")
    
    def __del__(self):
        """析構函數"""
        self.close()