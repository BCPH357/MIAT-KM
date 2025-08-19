#!/usr/bin/env python3
"""
重置向量數據庫腳本
清除舊的向量數據，為新的BGE-M3模型做準備
"""

import os
import shutil
import logging
from config import CHROMA_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_vector_database():
    """重置向量數據庫"""
    try:
        if os.path.exists(CHROMA_DB_PATH):
            logger.info(f"刪除現有向量數據庫: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info("向量數據庫刪除成功")
        else:
            logger.info("向量數據庫目錄不存在，無需刪除")
        
        # 重新創建空目錄
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        logger.info(f"重新創建向量數據庫目錄: {CHROMA_DB_PATH}")
        
        logger.info("向量數據庫重置完成！現在可以使用新的BGE-M3模型")
        
    except Exception as e:
        logger.error(f"重置向量數據庫失敗: {e}")

if __name__ == "__main__":
    print("重置向量數據庫以支援BGE-M3模型")
    confirm = input("確定要刪除所有現有向量數據嗎？(y/N): ")
    
    if confirm.lower() == 'y':
        reset_vector_database()
    else:
        print("取消重置操作")