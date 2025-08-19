#!/usr/bin/env python3
"""
Vector RAG處理調試腳本
用於追蹤文檔處理過程中卡住的具體步驟
"""

import os
import sys
import time
import logging
from datetime import datetime

# 設置詳細日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_file_processing():
    """調試文檔處理流程"""
    logger.info("開始調試文檔處理流程")
    
    try:
        # 檢查文件
        from config import PDF_DIR, MARKDOWN_DIR
        logger.info(f"PDF目錄: {PDF_DIR}")
        logger.info(f"Markdown目錄: {MARKDOWN_DIR}")
        
        # 掃描文件
        all_files = []
        
        if os.path.exists(MARKDOWN_DIR):
            for file_name in os.listdir(MARKDOWN_DIR):
                if file_name.lower().endswith(('.md', '.markdown')):
                    file_path = os.path.join(MARKDOWN_DIR, file_name)
                    all_files.append(file_path)
                    logger.info(f"找到Markdown文件: {file_path}")
        
        logger.info(f"總文件數: {len(all_files)}")
        
        if not all_files:
            logger.warning("沒有找到任何文件")
            return
        
        # 測試文檔分塊器
        logger.info("=== 測試文檔分塊器 ===")
        from document_chunker import DocumentChunker
        
        chunker = DocumentChunker()
        logger.info("文檔分塊器初始化成功")
        
        # 處理第一個文件
        test_file = all_files[0]
        logger.info(f"測試文件: {test_file}")
        
        logger.info("步驟1: 讀取文件內容...")
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"文件大小: {len(content)} 字符")
        
        logger.info("步驟2: 開始文檔分塊...")
        start_time = time.time()
        chunks = chunker.chunk_file(test_file)
        chunk_time = time.time() - start_time
        logger.info(f"文檔分塊完成: {len(chunks)} 個chunks，耗時 {chunk_time:.2f}秒")
        
        if chunks:
            first_chunk = chunks[0]
            logger.info(f"第一個chunk: {first_chunk['text'][:100]}...")
        
        # 測試embedding
        logger.info("=== 測試Embedding ===")
        from vector_embedder import QwenEmbedder
        
        logger.info("初始化embedding模型...")
        start_time = time.time()
        embedder = QwenEmbedder()
        init_time = time.time() - start_time
        logger.info(f"Embedding模型初始化完成，耗時 {init_time:.2f}秒")
        
        # 測試少量文本編碼
        test_texts = [chunk['text'] for chunk in chunks[:2]]  # 只測試前2個chunks
        logger.info(f"測試編碼 {len(test_texts)} 個文本...")
        
        start_time = time.time()
        embeddings = embedder.encode(test_texts)
        encode_time = time.time() - start_time
        logger.info(f"編碼完成，耗時 {encode_time:.2f}秒")
        
        # 測試向量存儲
        logger.info("=== 測試向量存儲 ===")
        from vector_retriever import VectorRetriever
        
        retriever = VectorRetriever()
        logger.info("向量檢索器初始化成功")
        
        # 準備測試數據
        ids = [f"test_{i}" for i in range(len(test_texts))]
        metadatas = [chunk['metadata'] for chunk in chunks[:2]]
        
        logger.info("測試存儲到向量數據庫...")
        start_time = time.time()
        success = retriever.add_documents(
            documents=test_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        store_time = time.time() - start_time
        logger.info(f"存儲{'成功' if success else '失敗'}，耗時 {store_time:.2f}秒")
        
        logger.info("調試完成!")
        
    except Exception as e:
        logger.error(f"調試過程中發生錯誤: {e}")
        logger.error(f"錯誤類型: {type(e).__name__}")
        import traceback
        logger.error(f"完整錯誤:\n{traceback.format_exc()}")

def debug_step_by_step():
    """逐步調試VectorRAGProcessor"""
    logger.info("開始逐步調試VectorRAGProcessor")
    
    try:
        logger.info("步驟1: 初始化各個組件...")
        
        # 1. 測試embedder
        logger.info("1.1 初始化QwenEmbedder...")
        from vector_embedder import QwenEmbedder
        embedder = QwenEmbedder()
        logger.info("✓ QwenEmbedder初始化成功")
        
        # 2. 測試chunker
        logger.info("1.2 初始化DocumentChunker...")
        from document_chunker import DocumentChunker
        chunker = DocumentChunker()
        logger.info("✓ DocumentChunker初始化成功")
        
        # 3. 測試retriever
        logger.info("1.3 初始化VectorRetriever...")
        from vector_retriever import VectorRetriever
        retriever = VectorRetriever()
        logger.info("✓ VectorRetriever初始化成功")
        
        logger.info("步驟2: 初始化VectorRAGProcessor...")
        from vector_rag_processor import VectorRAGProcessor
        processor = VectorRAGProcessor()
        logger.info("✓ VectorRAGProcessor初始化成功")
        
        logger.info("步驟3: 掃描文件...")
        from config import PDF_DIR, MARKDOWN_DIR
        
        all_files = []
        if os.path.exists(MARKDOWN_DIR):
            for file_name in os.listdir(MARKDOWN_DIR):
                if file_name.lower().endswith(('.md', '.markdown')):
                    all_files.append(os.path.join(MARKDOWN_DIR, file_name))
        
        logger.info(f"找到 {len(all_files)} 個文件")
        
        if not all_files:
            logger.warning("沒有文件要處理")
            return
        
        logger.info("步驟4: 處理單個文件...")
        test_file = all_files[0]
        logger.info(f"處理文件: {os.path.basename(test_file)}")
        
        start_time = time.time()
        result = processor.process_single_file(test_file)
        process_time = time.time() - start_time
        
        logger.info(f"單文件處理完成，耗時 {process_time:.2f}秒")
        logger.info(f"處理結果: {result}")
        
        logger.info("逐步調試完成!")
        
    except Exception as e:
        logger.error(f"逐步調試中發生錯誤: {e}")
        import traceback
        logger.error(f"完整錯誤:\n{traceback.format_exc()}")

if __name__ == "__main__":
    print("Vector RAG處理調試工具")
    print("1. 組件級別調試")
    print("2. 逐步調試")
    
    choice = input("選擇調試模式 (1/2): ")
    
    if choice == "1":
        debug_file_processing()
    elif choice == "2":
        debug_step_by_step()
    else:
        print("無效選擇")