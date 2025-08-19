#!/bin/bash

# Embedding模型測試執行腳本
# 用於在Docker容器中執行診斷測試

echo "=== MIAT-KM Embedding模型診斷 ==="
echo "執行時間: $(date)"
echo "=================================="

# 檢查Docker服務狀態
echo "1. 檢查Docker服務狀態..."
docker-compose ps

echo -e "\n2. 檢查容器資源使用..."
docker stats --no-stream

echo -e "\n3. 檢查GPU狀態 (如果可用)..."
docker-compose exec app nvidia-smi || echo "NVIDIA GPU不可用"

echo -e "\n4. 檢查Python依賴..."
docker-compose exec app python -c "
import sys
print(f'Python版本: {sys.version}')

packages = ['torch', 'transformers', 'sentence_transformers', 'chromadb', 'numpy']
for pkg in packages:
    try:
        module = __import__(pkg)
        if hasattr(module, '__version__'):
            print(f'{pkg}: {module.__version__}')
        else:
            print(f'{pkg}: 已安裝 (無版本信息)')
    except ImportError:
        print(f'{pkg}: 未安裝')
"

echo -e "\n5. 執行Embedding模型診斷測試..."
echo "注意: 此測試可能需要幾分鐘時間，請耐心等待..."

# 執行主要測試腳本
docker-compose exec app python test_embedding_model.py

echo -e "\n6. 檢查測試日誌..."
if docker-compose exec app test -f /app/data/embedding_test.log; then
    echo "測試日誌內容:"
    docker-compose exec app tail -50 /app/data/embedding_test.log
else
    echo "找不到測試日誌文件"
fi

echo -e "\n7. 檢查模型緩存狀態..."
docker-compose exec app du -sh /root/.cache/huggingface/ 2>/dev/null || echo "模型緩存目錄不存在或為空"

echo -e "\n診斷完成!"
echo "請檢查上述輸出以找出Vector RAG處理中斷的原因"