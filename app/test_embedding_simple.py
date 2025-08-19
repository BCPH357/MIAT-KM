#!/usr/bin/env python3
"""
簡化版embedding模型測試腳本（無需psutil）
用於快速檢測Vector RAG處理問題
"""

import os
import sys
import time
import torch
import logging
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_info():
    """測試基本環境信息"""
    logger.info("=== 基本環境信息 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"當前工作目錄: {os.getcwd()}")
    
    # CUDA信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA可用，GPU數量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {gpu_name}")
    else:
        logger.info("CUDA不可用，將使用CPU")

def test_dependencies():
    """測試依賴模組"""
    logger.info("=== 測試依賴模組 ===")
    
    modules = [
        'torch', 'transformers', 'sentence_transformers', 
        'chromadb', 'numpy', 'huggingface_hub'
    ]
    
    for module_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✓ {module_name}: {version}")
        except ImportError as e:
            logger.error(f"✗ {module_name}: {e}")

def test_huggingface_connection():
    """測試Hugging Face連接"""
    logger.info("=== 測試Hugging Face連接 ===")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # 測試連接
        logger.info("測試Hugging Face Hub連接...")
        model_info = api.model_info("Qwen/Qwen3-Embedding-8B")
        logger.info(f"✓ 連接成功，模型存在: {model_info.modelId}")
        logger.info(f"模型大小: {model_info.safetensors}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hugging Face連接失敗: {e}")
        return False

def test_model_download():
    """測試模型下載（僅tokenizer）"""
    logger.info("=== 測試模型下載 ===")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "Qwen/Qwen3-Embedding-8B"
        logger.info(f"嘗試下載tokenizer: {model_name}")
        
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        download_time = time.time() - start_time
        
        logger.info(f"✓ Tokenizer下載成功 ({download_time:.2f}秒)")
        logger.info(f"詞彙表大小: {len(tokenizer.get_vocab())}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Tokenizer下載失敗: {e}")
        return False

def test_fallback_model():
    """測試備用模型"""
    logger.info("=== 測試備用模型 ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("測試sentence-transformers備用模型...")
        start_time = time.time()
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        load_time = time.time() - start_time
        
        logger.info(f"✓ 備用模型載入成功 ({load_time:.2f}秒)")
        
        # 測試編碼
        test_text = "測試文本"
        embedding = model.encode([test_text])
        logger.info(f"✓ 編碼測試成功，向量維度: {embedding.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 備用模型失敗: {e}")
        return False

def test_vector_embedder_init():
    """測試VectorEmbedder初始化"""
    logger.info("=== 測試VectorEmbedder初始化 ===")
    
    try:
        sys.path.append('/app')
        
        logger.info("導入VectorEmbedder...")
        from vector_embedder import QwenEmbedder
        
        logger.info("開始初始化QwenEmbedder...")
        start_time = time.time()
        
        # 設置較短的超時時間來測試
        embedder = QwenEmbedder()
        
        init_time = time.time() - start_time
        logger.info(f"✓ QwenEmbedder初始化成功 ({init_time:.2f}秒)")
        
        # 檢查是否使用備用模型
        if hasattr(embedder, 'use_fallback'):
            logger.info("注意: 使用了備用模型")
        else:
            logger.info("使用了主要的Qwen模型")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ VectorEmbedder初始化失敗: {e}")
        logger.error(f"錯誤類型: {type(e).__name__}")
        return False

def test_config_values():
    """檢查配置值"""
    logger.info("=== 檢查配置值 ===")
    
    try:
        sys.path.append('/app')
        from config import (
            EMBEDDING_MODEL, EMBEDDING_DEVICE, 
            EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_LENGTH
        )
        
        logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
        logger.info(f"EMBEDDING_DEVICE: {EMBEDDING_DEVICE}")
        logger.info(f"EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}")
        logger.info(f"EMBEDDING_MAX_LENGTH: {EMBEDDING_MAX_LENGTH}")
        
    except Exception as e:
        logger.error(f"配置讀取失敗: {e}")

def main():
    """主測試函數"""
    logger.info(f"開始簡化診斷測試 - {datetime.now()}")
    logger.info("=" * 50)
    
    tests = [
        ("基本環境", test_basic_info),
        ("依賴模組", test_dependencies),
        ("配置檢查", test_config_values),
        ("HF連接", test_huggingface_connection),
        ("模型下載", test_model_download),
        ("備用模型", test_fallback_model),
        ("VectorEmbedder", test_vector_embedder_init)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*10} {test_name} {'='*10}")
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            results.append((test_name, success, duration))
            
            if success:
                logger.info(f"✓ {test_name} 成功 ({duration:.2f}秒)")
            else:
                logger.info(f"✗ {test_name} 失敗 ({duration:.2f}秒)")
                
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            logger.error(f"✗ {test_name} 異常: {e} ({duration:.2f}秒)")
    
    # 測試總結
    logger.info(f"\n{'='*20} 測試總結 {'='*20}")
    for test_name, success, duration in results:
        status = "✓" if success else "✗"
        logger.info(f"{status} {test_name}: {duration:.2f}秒")
    
    successful_tests = sum(1 for _, success, _ in results if success)
    logger.info(f"\n總計: {successful_tests}/{len(results)} 測試通過")
    logger.info("診斷完成!")

if __name__ == "__main__":
    main()