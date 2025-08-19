#!/usr/bin/env python3
"""
逐步調試Vector RAG處理流程
每個步驟都有詳細日誌和超時檢測
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime

# 設置詳細日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("操作超時")

def with_timeout(func, timeout_seconds=60, description="操作"):
    """執行函數並設置超時"""
    logger.info(f"開始{description}（超時: {timeout_seconds}秒）...")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        start_time = time.time()
        result = func()
        duration = time.time() - start_time
        logger.info(f"✓ {description}完成，耗時: {duration:.2f}秒")
        return result
    except TimeoutException:
        logger.error(f"✗ {description}超時（{timeout_seconds}秒）")
        return None
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ {description}失敗: {e}，耗時: {duration:.2f}秒")
        return None
    finally:
        signal.alarm(0)

def step1_scan_files():
    """步驟1: 掃描文件"""
    from config import MARKDOWN_DIR
    
    files = []
    if os.path.exists(MARKDOWN_DIR):
        for file_name in os.listdir(MARKDOWN_DIR):
            if file_name.lower().endswith(('.md', '.markdown')):
                files.append(os.path.join(MARKDOWN_DIR, file_name))
    
    logger.info(f"找到 {len(files)} 個Markdown文件")
    return files

def step2_init_embedder():
    """步驟2: 初始化BGE embedder"""
    from vector_embedder import BGEEmbedder
    return BGEEmbedder()

def step3_init_chunker():
    """步驟3: 初始化文檔分塊器"""
    from document_chunker import DocumentChunker
    return DocumentChunker()

def step4_init_retriever():
    """步驟4: 初始化向量檢索器"""
    from vector_retriever import VectorRetriever
    return VectorRetriever()

def step5_clear_database(retriever):
    """步驟5: 清空數據庫"""
    return retriever.clear_collection()

def step6_read_file(file_path):
    """步驟6: 讀取文件內容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    logger.info(f"文件大小: {len(content)} 字符")
    return content

def step7_chunk_file(chunker, file_path):
    """步驟7: 文檔分塊"""
    chunks = chunker.chunk_file(file_path)
    logger.info(f"生成 {len(chunks)} 個chunks")
    return chunks

def step8_encode_first_chunk(embedder, chunk):
    """步驟8: 編碼第一個chunk"""
    text = chunk['text'][:200] + "..."  # 只取前200字符測試
    embedding = embedder.encode_single(text)
    logger.info(f"向量維度: {len(embedding)}")
    return embedding

def step9_encode_batch(embedder, chunks, batch_size=2):
    """步驟9: 批次編碼（小批次測試）"""
    texts = [chunk['text'] for chunk in chunks[:batch_size]]
    embeddings = embedder.encode_batch(texts)
    logger.info(f"批次編碼完成: {len(embeddings)} 個向量")
    return embeddings

def step10_store_to_db(retriever, texts, embeddings, metadatas):
    """步驟10: 存儲到數據庫"""
    ids = [f"test_{i}" for i in range(len(texts))]
    success = retriever.add_documents(texts, embeddings, metadatas, ids)
    logger.info(f"數據庫存儲: {'成功' if success else '失敗'}")
    return success

def main():
    """主要調試流程"""
    logger.info("=" * 50)
    logger.info("開始逐步調試Vector RAG處理流程")
    logger.info("=" * 50)
    
    # 步驟1: 掃描文件
    files = with_timeout(step1_scan_files, 10, "掃描文件")
    if not files:
        logger.error("沒有找到文件，退出")
        return
    
    test_file = files[0]
    logger.info(f"測試文件: {os.path.basename(test_file)}")
    
    # 步驟2: 初始化embedder
    embedder = with_timeout(step2_init_embedder, 120, "初始化BGE embedder")
    if not embedder:
        logger.error("Embedder初始化失敗，退出")
        return
    
    # 步驟3: 初始化chunker
    chunker = with_timeout(step3_init_chunker, 10, "初始化文檔分塊器")
    if not chunker:
        logger.error("Chunker初始化失敗，退出")
        return
    
    # 步驟4: 初始化retriever
    retriever = with_timeout(step4_init_retriever, 30, "初始化向量檢索器")
    if not retriever:
        logger.error("Retriever初始化失敗，退出")
        return
    
    # 步驟5: 清空數據庫
    clear_result = with_timeout(lambda: step5_clear_database(retriever), 30, "清空數據庫")
    
    # 步驟6: 讀取文件
    content = with_timeout(lambda: step6_read_file(test_file), 10, "讀取文件")
    if not content:
        logger.error("文件讀取失敗，退出")
        return
    
    # 步驟7: 文檔分塊
    chunks = with_timeout(lambda: step7_chunk_file(chunker, test_file), 30, "文檔分塊")
    if not chunks:
        logger.error("文檔分塊失敗，退出")
        return
    
    # 步驟8: 編碼單個chunk測試
    first_embedding = with_timeout(
        lambda: step8_encode_first_chunk(embedder, chunks[0]), 
        60, "編碼單個chunk"
    )
    if first_embedding is None:
        logger.error("單個chunk編碼失敗，退出")
        return
    
    # 步驟9: 批次編碼測試（只測試2個chunks）
    batch_embeddings = with_timeout(
        lambda: step9_encode_batch(embedder, chunks, 2), 
        120, "批次編碼測試"
    )
    if not batch_embeddings:
        logger.error("批次編碼失敗，退出")
        return
    
    # 步驟10: 數據庫存儲測試
    test_texts = [chunk['text'] for chunk in chunks[:2]]
    test_metadatas = [chunk['metadata'] for chunk in chunks[:2]]
    
    store_result = with_timeout(
        lambda: step10_store_to_db(retriever, test_texts, batch_embeddings, test_metadatas),
        30, "數據庫存儲測試"
    )
    
    logger.info("=" * 50)
    logger.info("調試完成！")
    logger.info("=" * 50)
    
    if store_result:
        logger.info("✓ 所有步驟都成功完成，Vector RAG處理流程正常")
        
        # 測試完整流程
        logger.info("現在測試完整的文檔處理...")
        
        def full_process():
            from vector_rag_processor import VectorRAGProcessor
            processor = VectorRAGProcessor()
            return processor.process_single_file(test_file)
        
        full_result = with_timeout(full_process, 300, "完整文檔處理")
        if full_result:
            logger.info("✓ 完整文檔處理成功")
            logger.info(f"處理結果: {full_result}")
        else:
            logger.error("✗ 完整文檔處理失敗或超時")
    else:
        logger.error("✗ 某些步驟失敗，需要進一步檢查")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用戶中斷")
    except Exception as e:
        logger.error(f"調試過程中發生未預期錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())