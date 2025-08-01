# MIAT-KM: Neo4j RAG 知識管理系統

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

一個整合 **Neo4j 知識圖譜**、**LangChain** 和 **Ollama** 的智能 RAG (Retrieval-Augmented Generation) 系統。支援從 PDF 文件自動提取知識三元組，建構知識圖譜，並提供基於 LLM 的智能問答功能。

## ✨ 主要功能

- 🔄 **自動三元組提取**: 使用 Ollama Gemma3 12B 從 PDF 文件提取知識三元組
- 📊 **知識圖譜構建**: 將提取的三元組自動導入 Neo4j 建構知識圖譜
- 🤖 **智能問答**: 基於 LangChain + Neo4j + Ollama 的 RAG 系統
- ⚡ **GPU 加速**: 支援 NVIDIA GPU 加速 LLM 推理
- 🐳 **容器化部署**: 完整的 Docker Compose 一鍵部署方案

## 🏗️ 系統架構

```
用戶查詢 → LangChain → 生成Cypher查詢 → Neo4j → 檢索知識 → Ollama → 生成回答
```

## 📁 專案結構

```
MIAT-KM/
├── docker-compose.yml           # Docker 服務編排
├── README.md                    # 專案說明文件
├── .gitignore                   # Git 忽略規則
└── app/                         # 應用程式目錄
    ├── Dockerfile               # Python 應用容器配置
    ├── requirements.txt         # Python 依賴套件
    ├── main.py                  # 主程式入口點
    ├── sentence_triplet_extractor.py  # 三元組提取器
    ├── import_to_neo4j.py       # Neo4j 數據導入工具
    ├── rag_system.py            # RAG 系統核心邏輯
    ├── knowledge_retriever.py   # 知識檢索器 (含 LangChain 整合)
    ├── ollama_client.py         # Ollama API 客戶端
    └── data/                    # 數據目錄
        ├── pdf/                 # PDF 文件存放處
        └── processed/           # 處理後數據存放處
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
   
   # 下載 Gemma3 12B 模型 (推薦)
   ollama pull gemma3:12b
   
   # 退出容器
   exit
   ```

4. **驗證服務狀態**
   ```bash
   sudo docker-compose ps
   ```

### 🎯 使用方式

1. **放置 PDF 文件**
   ```bash
   # 將你的 PDF 文件複製到 app/data/pdf/ 目錄
   cp your-document.pdf app/data/pdf/
   ```

2. **啟動主程式**
   ```bash
   sudo docker-compose exec app python main.py
   ```

3. **選擇功能**
   ```
   === 知識圖譜應用菜單 ===
   1. 從 PDF 文件提取三元組
   2. 將三元組導入到 Neo4j
   3. RAG 問答系統
   4. 退出
   ```

4. **使用 RAG 問答**
   - 直接輸入問題：系統會自動使用 LangChain 進行智能問答
   - 輸入 `langchain <問題>`：顯示詳細的檢索過程
   - 輸入 `quit` 或 `exit`：退出問答系統

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
```

## 📁 核心文件說明

### 🔧 配置文件
- **`docker-compose.yml`**: 定義 Neo4j、Ollama、App 三個服務的配置
- **`app/Dockerfile`**: Python 應用的容器構建配置
- **`app/requirements.txt`**: Python 依賴套件清單

### 🎯 主程式
- **`main.py`**: 系統主入口，提供用戶交互菜單
- **`rag_system.py`**: RAG 系統核心邏輯，整合知識檢索和回答生成

### 🔍 知識處理
- **`sentence_triplet_extractor.py`**: 使用 Ollama 從 PDF 逐句提取三元組
- **`import_to_neo4j.py`**: 將 CSV 格式的三元組批量導入 Neo4j
- **`knowledge_retriever.py`**: 知識檢索器，整合 LangChain GraphCypherQAChain

### 🤖 LLM 整合
- **`ollama_client.py`**: Ollama API 客戶端，處理與本地 LLM 的通信

## 🌐 服務訪問

- **Neo4j 瀏覽器**: http://localhost:7475
  - 用戶名: `neo4j`
  - 密碼: `password123`
- **Ollama API**: http://localhost:11435
- **應用程式**: 通過 Docker Compose 執行

## ⚙️ 系統配置

### 💾 硬體需求
- **最低配置**: 8GB RAM, 4 CPU cores
- **推薦配置**: 16GB+ RAM, 8+ CPU cores, NVIDIA GPU (8GB+ VRAM)
- **磁碟空間**: 至少 20GB 可用空間

### 🔧 環境變數
可在 `docker-compose.yml` 中調整：
- Neo4j 記憶體配置: `NEO4J_dbms_memory_*`
- Ollama 模型路徑: `/root/.ollama` 卷掛載
- 應用數據路徑: `./app/data` 卷掛載

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

### 🔄 數據備份
```bash
# 備份 Neo4j 數據
sudo docker-compose exec neo4j neo4j-admin database dump neo4j /data/neo4j.dump

# 備份處理後的數據
cp -r app/data/processed/ backup/
```

## 🤝 貢獻指南

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 創建 Pull Request

## 📄 授權協議

此專案採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 文件

## 📧 聯繫方式

- 專案維護者: BCPH357
- Email: billy920225@gmail.com
- GitHub: [@BCPH357](https://github.com/BCPH357)

---

**🎉 快速開始你的知識管理之旅！** 如果遇到任何問題，請查看 Issues 或創建新的 Issue。 