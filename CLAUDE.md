# MIAT-KM: 雙重RAG知識管理系統

**快速理解**: Neo4j知識圖譜 + Vector RAG的雙重檢索系統，支持PDF/Markdown處理，使用QWEN3-Embedding和ChromaDB。

## 🏗️ 系統架構

### Docker服務
- **Neo4j**: 知識圖譜數據庫 (UI: 7475, Bolt: 7688, neo4j/password123)
- **Ollama**: 本地LLM服務 (11435端口, GPU加速)
- **App**: Python主應用 (包含所有邏輯)

### 核心功能模塊
```
main.py                    # 5選項交互菜單
├── 1. 三元組提取           # sentence_triplet_extractor.py
├── 2. 導入Neo4j           # import_to_neo4j.py  
├── 3. RAG問答             # rag_system.py (多模式)
├── 4. Vector RAG預處理    # vector_rag_processor.py
└── 5. 退出

config.py                  # 全局配置中心
knowledge_retriever.py     # Neo4j + LangChain檢索
ollama_client.py          # LLM通信
```

### Vector RAG組件 (新增)
```
vector_embedder.py        # QWEN3-Embedding-8B封裝
document_chunker.py       # 智能文檔分塊
vector_retriever.py       # ChromaDB客戶端  
vector_rag_processor.py   # 整合處理流程
```

## ⚡ 快速開始

### 1. 啟動系統
```bash
sudo docker-compose up -d
sudo docker-compose exec app python main.py
```

### 2. 處理文檔
```bash
# 將PDF/MD文件放入對應目錄
./app/data/pdf/
./app/data/markdown/

# 選項1: 提取三元組
# 選項2: 導入Neo4j  
# 選項4: Vector RAG預處理 (新增)
```

### 3. RAG問答 (選項3)
```bash
# 直接輸入問題 → 知識圖譜LangChain
# hybrid <問題> → 圖譜混合RAG
# vector <問題> → 純向量RAG
# hybrid-all <問題> → 雙重RAG (推薦)
# compare <問題> → 比較所有模式
```

## 🔧 關鍵配置

### config.py 重要設定
```python
# LLM配置
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://ollama:11434"

# Vector Embedding配置
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"  # 專業embedding模型
EMBEDDING_DEVICE = "cuda"                      # GPU加速
CHROMA_DB_PATH = "/app/data/chroma_db"        # 向量數據庫

# 數據目錄
PDF_DIR = "/app/data/pdf"
MARKDOWN_DIR = "/app/data/markdown"
```

### Docker Volumes持久化
```yaml
volumes:
  - neo4j_data:/data                    # Neo4j數據
  - ollama_data:/root/.ollama           # LLM模型
  - huggingface_cache:/root/.cache      # Embedding模型緩存
  - ./app/data/chroma_db:/app/data/chroma_db  # 向量數據庫
```

## 🚀 RAG模式對比

| 模式 | 檢索方式 | 適用場景 | 命令 |
|------|----------|----------|------|
| **知識圖譜** | 結構化關係 | 邏輯推理、實體關係 | 直接輸入 |
| **向量RAG** | 語義相似度 | 內容檢索、模糊匹配 | `vector <問題>` |
| **雙重RAG** | 圖譜+向量 | 全面檢索、最佳效果 | `hybrid-all <問題>` |

## 🛠️ 開發工作流

### 代碼修改
```bash
# 修改 ./app/ 下的Python文件 (實時映射)
sudo docker-compose restart app  # 重載變更
```

### 依賴更新
```bash
# 更新 app/requirements.txt
sudo docker-compose build app
sudo docker-compose up -d
```

### 問題排除
```bash
sudo docker-compose logs app          # 查看應用日誌
sudo docker-compose exec app bash     # 進入容器調試
```

## 📊 數據流程

### 文檔 → 知識圖譜
1. PDF/MD → 三元組提取 (LLM)
2. 三元組 → Neo4j導入
3. 問答 → Cypher查詢檢索

### 文檔 → 向量數據庫
1. PDF/MD → 智能分塊 (語義完整)
2. 分塊 → QWEN3-Embedding
3. 向量 → ChromaDB存儲
4. 問答 → 餘弦相似度檢索

## 🔒 Git規範

- **目標分支**: 始終推送到 `main`
- **提交格式**: 簡潔描述，無co-author標籤
- **示例**: `git commit -m "Add vector RAG functionality"`

## 💡 系統特色

✅ **雙重檢索**: 知識圖譜 + 向量檢索互補  
✅ **多語言**: 支持中英文PDF/Markdown  
✅ **GPU加速**: QWEN3-Embedding + Ollama  
✅ **持久化**: 模型緩存避免重複下載  
✅ **交互式**: 5選項菜單 + 多模式問答  
✅ **可擴展**: 模塊化架構，易於擴展