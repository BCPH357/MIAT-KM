# MIAT-KM: 雙重RAG知識管理系統

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

一個整合 **Neo4j 知識圖譜**、**BGE-M3 向量檢索** 和 **Ollama** 的雙重RAG系統。支援從 PDF/Markdown 文件自動提取知識三元組、建構知識圖譜、向量數據庫，並提供三種模式的智能問答功能。

## ✨ 主要功能

- 🔄 **自動三元組提取**: 使用 Ollama LLM 從 PDF/Markdown 文件提取知識三元組
- 📊 **知識圖譜構建**: 將提取的三元組自動導入 Neo4j 建構知識圖譜
- 🔍 **向量檢索系統**: 使用 BGE-M3 embedding 模型 + ChromaDB 建構向量檢索系統
- 🤖 **三模式智能問答**: 
  - **KG模式**: 基於Neo4j知識圖譜的結構化檢索
  - **Vector模式**: 基於BGE-M3 + ChromaDB的語義相似度檢索
  - **Hybrid-All模式**: 融合知識圖譜與向量檢索的雙重RAG系統
- 🆚 **模式比較功能**: 即時比較三種問答模式的效果差異
- ⚡ **GPU 加速**: 支援 NVIDIA GPU 加速 LLM 推理 (embedding使用CPU避免GPU記憶體衝突)
- 🐳 **容器化部署**: 完整的 Docker Compose 一鍵部署方案

## 🏗️ 系統架構

```
        ┌─────────────────────────────────────────────────────────┐
        │                 用戶查詢輸入                           │
        └─────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────▼───────────────────────────────────────┐
        │                選擇RAG模式                            │ 
        └─┬─────────────┬─────────────┬─────────────────────────┘
          │             │             │
     ┌────▼───┐   ┌─────▼────┐   ┌────▼────────┐
     │KG模式  │   │Vector模式│   │Hybrid-All   │
     │        │   │          │   │模式         │
     └────┬───┘   └─────┬────┘   └────┬────────┘
          │             │             │
    ┌─────▼─────┐ ┌─────▼─────┐      │
    │Neo4j圖譜  │ │BGE-M3     │      │
    │Cypher查詢 │ │+ChromaDB  │      │
    │          │ │向量檢索    │      │
    └─────┬─────┘ └─────┬─────┘      │
          │             │             │
          └──────┬──────┴─────────────┤
                 │                    │
          ┌──────▼──────┐    ┌───────▼──────┐
          │  檢索結果   │    │ 雙重檢索結合 │
          │    整合     │    │   (Hybrid)   │
          └──────┬──────┘    └───────┬──────┘
                 │                    │
                 └──────┬─────────────┘
                        │
              ┌─────────▼─────────┐
              │   Ollama LLM     │
              │   生成最終回答    │
              └───────────────────┘
```

## 📁 專案結構

```
MIAT-KM/
├── docker-compose.yml           # Docker 服務編排 (Neo4j + Ollama + App)
├── README.md                    # 專案說明文件
├── CLAUDE.md                    # 專案快速指南
├── .gitignore                   # Git 忽略規則
└── app/                         # 應用程式目錄
    ├── Dockerfile               # Python 應用容器配置
    ├── requirements.txt         # Python 依賴套件
    ├── config.py                # 全域配置中心
    ├── main.py                  # 主程式入口點
    │
    ├── # 知識圖譜相關
    ├── sentence_triplet_extractor.py  # 三元組提取器
    ├── import_to_neo4j.py              # Neo4j 數據導入工具
    ├── knowledge_retriever.py          # 知識圖譜檢索器
    │
    ├── # 向量RAG相關
    ├── vector_embedder.py              # BGE-M3 向量編碼器
    ├── vector_retriever.py             # ChromaDB 向量檢索器
    ├── document_chunker.py             # 智能文檔分塊器
    ├── vector_rag_processor.py         # 向量RAG處理器
    │
    ├── # 系統核心
    ├── rag_system.py                   # 三模式RAG系統整合
    ├── ollama_client.py                # Ollama API 客戶端
    │
    ├── # 實用工具
    ├── reset_vector_db.py              # 向量數據庫重置工具
    │
    └── data/                           # 數據目錄
        ├── pdf/                        # PDF 文件存放處
        ├── markdown/                   # Markdown 文件存放處
        ├── processed/                  # 處理後數據存放處
        ├── chroma_db/                  # ChromaDB 向量數據庫
        └── vector_db/                  # 其他向量數據
```

## 🚀 快速開始

### 📋 前置需求

#### 1. 安裝 Docker 和 Docker Compose
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### 2. GPU 加速支援 (可選，但強烈建議)

如果你有 NVIDIA GPU，安裝以下工具包以啟用 GPU 加速：

```bash
# Ubuntu/Debian 系統
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 配置 Docker 使用 NVIDIA 運行時
sudo systemctl restart docker

# 驗證安裝
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### 🔧 部署步驟

1. **克隆專案**
   ```bash
   git clone https://github.com/BCPH357/MIAT-KM.git
   cd MIAT-KM
   ```

2. **啟動服務**
   ```bash
   sudo docker-compose up -d
   ```

3. **下載並安裝 LLM 模型**
   ```bash
   # 進入 Ollama 容器
   sudo docker-compose exec ollama bash
   
   # 下載 gpt-oss:20b 模型 (推薦)
   ollama pull gpt-oss:20b
   
   # 退出容器
   exit
   ```

4. **驗證服務狀態**
   ```bash
   sudo docker-compose ps
   ```

### 🎯 使用方式

#### 第一次使用需要準備數據

1. **放置文件**
   ```bash
   # 將你的 PDF 文件複製到 app/data/pdf/ 目錄
   cp your-document.pdf app/data/pdf/
   
   # 將你的 Markdown 文件複製到 app/data/markdown/ 目錄
   cp your-document.md app/data/markdown/
   ```

2. **啟動主程式並建立知識庫**
   ```bash
   sudo docker-compose exec app python main.py
   ```

3. **按照菜單順序執行**
   ```
   === 知識圖譜應用菜單 ===
   1. 從文件提取三元組 (PDF 和 Markdown)     # 先執行這個
   2. 將三元組導入到 Neo4j                    # 再執行這個
   3. RAG 問答系統                           # 問答功能
   4. Vector RAG 文檔預處理                   # 向量RAG準備
   5. 退出
   ```

#### 建立向量檢索系統

4. **Vector RAG 預處理 (選擇選項4)**
   ```
   === Vector RAG 文檔預處理 ===
   1. 處理所有文檔 (清空現有數據)              # 第一次選這個
   2. 處理所有文檔 (增量模式)                 # 後續新增文件選這個
   3. 查看數據庫統計
   4. 清空向量數據庫
   5. 測試向量搜索
   6. 返回主菜單
   ```

#### 使用三模式RAG問答

5. **使用 RAG 問答系統 (選擇選項3)**
   
   系統提供三種問答模式：
   
   - **`KG <問題>`**: 使用知識圖譜模式
   - **`vector <問題>`**: 使用純向量RAG模式  
   - **`hybrid-all <問題>`**: 使用全混合模式(知識圖譜+向量)
   - **`compare <問題>`**: 比較三種模式效果
   - **`quit` 或 `exit`**: 退出問答系統
   
   **使用範例**:
   ```
   請輸入問題: KG MIAT方法論是什麼
   請輸入問題: vector 如何設計系統架構
   請輸入問題: hybrid-all 什麼是離散事件建模
   請輸入問題: compare RAG系統的優勢
   ```

## 🔧 配置說明

### 系統核心配置 (config.py)

```python
# LLM模型配置
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://ollama:11434"

# Neo4j配置
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"  
NEO4J_PASSWORD = "password123"

# BGE-M3 Embedding配置
EMBEDDING_MODEL = "BAAI/bge-m3"          # 多語言embedding模型
EMBEDDING_DEVICE = "cpu"                 # 強制CPU避免GPU記憶體衝突
EMBEDDING_BATCH_SIZE = 16                # 批次處理大小
EMBEDDING_MAX_LENGTH = 512               # 最大序列長度

# ChromaDB配置
CHROMA_DB_PATH = "/app/data/chroma_db"
CHROMA_COLLECTION_NAME = "miat_documents"

# 文檔分塊配置
CHUNK_SIZE = 512                         # 每個chunk最大字符數
CHUNK_OVERLAP = 50                       # chunk重疊字符數
MIN_CHUNK_SIZE = 100                     # 最小chunk大小
```

### Docker容器配置

- **Neo4j**: 7475端口(Web UI), 7688端口(Bolt)
- **Ollama**: 11435端口, GPU加速
- **App**: Python應用, GPU加速, 掛載數據目錄
- **數據持久化**: 所有數據通過volumes持久保存

## 🆚 三種RAG模式比較

| 模式 | 檢索方式 | 適用場景 | 優勢 | 劣勢 |
|------|----------|----------|------|------|
| **KG** | 結構化Cypher查詢 | 邏輯推理、實體關係查詢 | 精確、可解釋 | 需要結構化知識 |
| **Vector** | 語義相似度匹配 | 內容檢索、模糊查詢 | 覆蓋面廣、語義理解 | 可能不夠精確 |
| **Hybrid-All** | 知識圖譜+向量雙重檢索 | 複雜問答、全面分析 | 結合兩者優勢 | 計算資源需求高 |

### 使用建議

- **日常查詢**: 使用 `KG <問題>` (快速、精確)
- **內容搜索**: 使用 `vector <問題>` (語義匹配)
- **重要問題**: 使用 `hybrid-all <問題>` (最全面)
- **效果比較**: 使用 `compare <問題>` (了解差異)

## 🐳 Docker Compose 常用命令

### 基本操作
```bash
# 啟動所有服務 (後台運行)
sudo docker-compose up -d

# 查看服務狀態
sudo docker-compose ps

# 查看服務日誌
sudo docker-compose logs [service_name]

# 停止所有服務
sudo docker-compose down

# 停止並移除所有容器、網路、卷
sudo docker-compose down -v
```

### 開發相關
```bash
# 重新構建特定服務
sudo docker-compose build app

# 重啟特定服務
sudo docker-compose restart app

# 進入服務容器
sudo docker-compose exec app bash
sudo docker-compose exec neo4j bash
sudo docker-compose exec ollama bash

# 查看實時日誌
sudo docker-compose logs -f app
```

### 故障排除
```bash
# 強制重新創建容器
sudo docker-compose up --force-recreate

# 清除所有 Docker 緩存 (謹慎使用)
sudo docker system prune -a

# 查看 GPU 使用情況 (需要 nvidia-docker)
sudo docker-compose exec ollama nvidia-smi

# 重置向量數據庫
sudo docker-compose exec app python reset_vector_db.py
```

## 🌐 服務訪問

- **Neo4j 瀏覽器**: http://localhost:7475
  - 用戶名: `neo4j`
  - 密碼: `password123`
- **Ollama API**: http://localhost:11435
- **應用程式**: 通過 Docker Compose 執行

## ⚙️ 系統要求

### 💾 硬體需求
- **最低配置**: 16GB RAM, 8 CPU cores
- **推薦配置**: 32GB+ RAM, 16+ CPU cores, NVIDIA GPU (16GB+ VRAM)
- **磁碟空間**: 至少 50GB 可用空間 (模型檔案較大)

### 📊 性能建議
- **embedding模型**: 使用CPU避免GPU記憶體衝突
- **LLM推理**: 使用GPU加速
- **文檔分塊**: 適當調整CHUNK_SIZE以平衡精度和效率
- **批次處理**: 根據記憶體大小調整EMBEDDING_BATCH_SIZE

## 🛠️ 開發指南

### 📝 修改代碼後更新
```bash
# 1. 修改 Python 代碼 (即時生效，因為有 volume 掛載)
# 2. 重啟 app 服務
sudo docker-compose restart app
```

### 📦 添加新依賴
```bash
# 1. 修改 requirements.txt
# 2. 重新構建容器
sudo docker-compose build app
# 3. 重啟服務
sudo docker-compose up -d
```

### 🔄 數據備份與恢復
```bash
# 備份 Neo4j 數據
sudo docker-compose exec neo4j neo4j-admin database dump neo4j /data/neo4j.dump

# 備份向量數據庫
cp -r app/data/chroma_db/ backup/chroma_db/

# 備份處理後的數據
cp -r app/data/processed/ backup/processed/
```

### 🔧 模型管理
```bash
# 查看已安裝模型
sudo docker-compose exec ollama ollama list

# 下載新模型
sudo docker-compose exec ollama ollama pull model_name

# 刪除模型
sudo docker-compose exec ollama ollama rm model_name
```

## 🚀 系統特色

### 🔄 雙重RAG架構
- **知識圖譜**: 結構化知識，精確推理
- **向量檢索**: 語義理解，覆蓋面廣  
- **智能融合**: 自動選擇最佳檢索方式

### 🎯 多語言支援
- **BGE-M3**: 中英文都有優秀表現
- **文檔處理**: 支援PDF、Markdown等格式
- **智能分塊**: 保持語義完整性

### ⚡ 性能優化
- **GPU/CPU合理分配**: LLM用GPU，embedding用CPU
- **模型緩存**: 避免重複下載
- **批次處理**: 提升向量化效率

## 🤝 貢獻指南

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 創建 Pull Request

## ❓ 常見問題

### Q: Vector RAG 處理卡住怎麼辦？
A: 檢查GPU記憶體使用情況，必要時重啟容器或清理GPU記憶體。

### Q: 如何更換embedding模型？
A: 修改 config.py 中的 EMBEDDING_MODEL，重建容器，並清空向量數據庫重新處理文檔。

### Q: Neo4j 連接失敗？
A: 確認服務已啟動，檢查密碼是否正確，必要時重新創建容器。

### Q: 模型下載速度慢？
A: 可以使用代理或鏡像加速，或手動下載模型檔案。

## 📄 授權協議

此專案採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 文件

## 📧 聯繫方式

- 專案維護者: BCPH357
- Email: billy920225@gmail.com
- GitHub: [@BCPH357](https://github.com/BCPH357)

---

**🎉 開始你的雙重RAG知識管理之旅！** 如果遇到任何問題，請查看 Issues 或創建新的 Issue。