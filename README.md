# RAG-Neo4j-Ollama

這是一個結合 Neo4j 知識圖譜和 Ollama 大型語言模型的 RAG (Retrieval-Augmented Generation) 應用示例。

## 專案結構

```
rag-neo4j-ollama/
├── docker-compose.yml    # Docker 組合配置
├── app/
│   ├── Dockerfile        # Python 應用的 Docker 配置
│   ├── requirements.txt  # Python 依賴
│   ├── main.py           # 主應用程式
│   ├── triple_extraction.py  # 三元組抽取腳本
│   ├── import_to_neo4j.py    # Neo4j 導入腳本
│   ├── process_pdfs.py       # PDF 處理腳本
│   └── data/
│       ├── pdf/          # 存放 PDF 文件
│       └── processed/    # 存放處理後的數據
```

## 技術棧

- **Neo4j**: 圖形數據庫，用於存儲和查詢知識圖譜
- **Ollama**: 本地運行大型語言模型的工具
- **Python**: 應用程式語言
- **Docker**: 容器化平台
- **spaCy**: 自然語言處理庫，用於三元組抽取
- **PyPDF2/pypdf**: PDF 文件處理庫

## 快速開始

### 前提條件

- 安裝 [Docker](https://www.docker.com/get-started)
- 安裝 [Docker Compose](https://docs.docker.com/compose/install/)

### 運行應用

1. 克隆此倉庫：
   ```bash
   git clone <repository-url>
   cd rag-neo4j-ollama
   ```

2. 啟動服務：
   ```bash
   docker-compose up
   ```

3. 訪問服務：
   - Neo4j 瀏覽器: http://localhost:7474 (用戶名: neo4j, 密碼: password123)
   - 應用程序輸出可在控制台中查看

## 使用三元組抽取功能

1. 將 PDF 文件放入 `app/data/pdf` 目錄中

2. 在應用程序菜單中選擇選項 2 提取三元組：
   ```
   === 知識圖譜應用菜單 ===
   1. 創建示例知識圖譜
   2. 從 PDF 文件提取三元組
   3. 將三元組導入到 Neo4j
   4. 查詢知識圖譜
   5. 退出
   ```

3. 提取完成後，選擇選項 3 將三元組導入到 Neo4j

4. 使用選項 4 或直接在 Neo4j 瀏覽器中查詢知識圖譜

## 使用 Ollama 模型

默認情況下，應用會檢查 Ollama 服務中可用的模型。如果需要下載特定模型，可以：

1. 進入 Ollama 容器：
   ```bash
   docker exec -it <ollama-container-id> /bin/bash
   ```

2. 下載模型：
   ```bash
   ollama pull llama2
   ```

## 自定義

- 修改 `app/main.py` 來自定義應用邏輯
- 修改 `app/triple_extraction.py` 來自定義三元組抽取邏輯
- 在 `docker-compose.yml` 中調整服務配置
- 在 `app/requirements.txt` 中添加所需的 Python 包

## 修改程式後如何更新？

### 修改 Python 程式碼：
- 由於你有 volume 掛載 (`./app:/app`)，直接修改本地 app 目錄中的檔案後，容器內會立即看到更改
- 修改後使用 `docker-compose restart app` 重啟 app 服務即可

### 添加新的 Python 依賴：
- 如果修改了 requirements.txt，需要重新構建容器：`docker-compose build app`
- 然後重新啟動：`docker-compose up -d`

### 修改 Docker 配置：
- 如果修改了 docker-compose.yml 或 Dockerfile，需要重新構建：`docker-compose up --build`

## 授權

MIT 