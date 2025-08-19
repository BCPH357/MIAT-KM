#!/usr/bin/env python3
"""
Embedding模型測試和診斷腳本
用於檢測Vector RAG處理時程式中斷的原因
"""

import os
import sys
import time
import torch
import psutil
import logging
from datetime import datetime

# 設置詳細日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/data/embedding_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def log_system_info():
    """記錄系統資源信息"""
    logger.info("=== 系統資源信息 ===")
    
    # CPU信息
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU核心數: {cpu_count}")
    logger.info(f"CPU使用率: {cpu_percent}%")
    
    # 記憶體信息
    memory = psutil.virtual_memory()
    logger.info(f"總記憶體: {memory.total / (1024**3):.2f} GB")
    logger.info(f"可用記憶體: {memory.available / (1024**3):.2f} GB")
    logger.info(f"記憶體使用率: {memory.percent}%")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"GPU數量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # GPU記憶體使用情況
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU {i} 已分配記憶體: {allocated:.2f} GB")
            logger.info(f"GPU {i} 緩存記憶體: {cached:.2f} GB")
    else:
        logger.info("CUDA不可用，將使用CPU")
    
    logger.info("========================\n")

def test_model_download():
    """測試模型下載"""
    logger.info("=== 測試模型下載 ===")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import snapshot_download
        import transformers
        
        logger.info(f"transformers版本: {transformers.__version__}")
        
        model_name = "Qwen/Qwen3-Embedding-8B"
        logger.info(f"目標模型: {model_name}")
        
        # 檢查緩存目錄
        cache_dir = "/root/.cache/huggingface"
        if os.path.exists(cache_dir):
            cache_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(cache_dir)
                for filename in filenames
            ) / (1024**3)
            logger.info(f"Hugging Face緩存大小: {cache_size:.2f} GB")
        else:
            logger.info("Hugging Face緩存目錄不存在")
        
        # 測試網路連接
        logger.info("測試Hugging Face Hub連接...")
        start_time = time.time()
        
        try:
            # 先測試tokenizer下載（較小）
            logger.info("下載tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            tokenizer_time = time.time() - start_time
            logger.info(f"Tokenizer下載成功 ({tokenizer_time:.2f}秒)")
            
            # 記錄當前記憶體使用
            log_system_info()
            
            # 嘗試下載模型
            logger.info("下載模型...")
            model_start = time.time()
            
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=torch.float16  # 使用半精度減少記憶體
            )
            
            model_time = time.time() - model_start
            logger.info(f"模型下載成功 ({model_time:.2f}秒)")
            
            # 測試模型加載到GPU
            if torch.cuda.is_available():
                logger.info("將模型移動到GPU...")
                gpu_start = time.time()
                model = model.cuda()
                gpu_time = time.time() - gpu_start
                logger.info(f"模型GPU加載成功 ({gpu_time:.2f}秒)")
                
                # 記錄GPU記憶體使用
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU記憶體使用: {allocated:.2f} GB")
            
            logger.info("模型測試完成!")
            return True
            
        except Exception as e:
            logger.error(f"模型下載/加載失敗: {e}")
            logger.error(f"錯誤類型: {type(e).__name__}")
            return False
            
    except ImportError as e:
        logger.error(f"缺少依賴模組: {e}")
        return False

def test_fallback_model():
    """測試備用模型"""
    logger.info("=== 測試備用模型 ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("測試sentence-transformers模型...")
        start_time = time.time()
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        load_time = time.time() - start_time
        logger.info(f"備用模型載入成功 ({load_time:.2f}秒)")
        
        # 測試編碼
        test_text = "這是一個測試文本"
        embedding = model.encode([test_text])
        logger.info(f"編碼測試成功，向量維度: {embedding.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"備用模型測試失敗: {e}")
        return False

def test_vector_embedder():
    """測試完整的VectorEmbedder類"""
    logger.info("=== 測試VectorEmbedder類 ===")
    
    try:
        sys.path.append('/app')
        from vector_embedder import QwenEmbedder
        
        logger.info("初始化QwenEmbedder...")
        start_time = time.time()
        
        embedder = QwenEmbedder()
        init_time = time.time() - start_time
        logger.info(f"QwenEmbedder初始化完成 ({init_time:.2f}秒)")
        
        # 測試編碼
        test_texts = ["這是第一個測試", "這是第二個測試"]
        logger.info("測試文本編碼...")
        
        encode_start = time.time()
        embeddings = embedder.encode(test_texts)
        encode_time = time.time() - encode_start
        
        logger.info(f"編碼完成 ({encode_time:.2f}秒)")
        logger.info(f"生成 {len(embeddings)} 個向量")
        
        if len(embeddings) > 0:
            logger.info(f"向量維度: {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        logger.error(f"VectorEmbedder測試失敗: {e}")
        logger.error(f"錯誤詳情: ", exc_info=True)
        return False

def test_document_processing():
    """測試文檔處理流程"""
    logger.info("=== 測試文檔處理流程 ===")
    
    try:
        sys.path.append('/app')
        from vector_rag_processor import VectorRAGProcessor
        
        logger.info("初始化VectorRAGProcessor...")
        start_time = time.time()
        
        processor = VectorRAGProcessor()
        init_time = time.time() - start_time
        logger.info(f"VectorRAGProcessor初始化完成 ({init_time:.2f}秒)")
        
        # 檢查數據庫狀態
        stats = processor.get_database_stats()
        logger.info(f"數據庫狀態: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"文檔處理測試失敗: {e}")
        logger.error(f"錯誤詳情: ", exc_info=True)
        return False

def main():
    """主測試函數"""
    logger.info(f"開始embedding模型診斷 - {datetime.now()}")
    logger.info("=" * 50)
    
    # 記錄初始系統信息
    log_system_info()
    
    results = {}
    
    # 執行各項測試
    tests = [
        ("模型下載測試", test_model_download),
        ("備用模型測試", test_fallback_model),
        ("VectorEmbedder測試", test_vector_embedder),
        ("文檔處理測試", test_document_processing)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            results[test_name] = {
                'success': result,
                'duration': duration
            }
            logger.info(f"{test_name} 完成: {'成功' if result else '失敗'} ({duration:.2f}秒)")
        except Exception as e:
            results[test_name] = {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }
            logger.error(f"{test_name} 發生異常: {e}")
    
    # 輸出測試總結
    logger.info(f"\n{'='*20} 測試總結 {'='*20}")
    for test_name, result in results.items():
        status = "✓ 成功" if result['success'] else "✗ 失敗"
        duration = result['duration']
        logger.info(f"{test_name}: {status} ({duration:.2f}秒)")
        if not result['success'] and 'error' in result:
            logger.info(f"  錯誤: {result['error']}")
    
    # 最終系統信息
    logger.info(f"\n{'='*20} 最終系統狀態 {'='*20}")
    log_system_info()
    
    logger.info("診斷完成!")

if __name__ == "__main__":
    main()