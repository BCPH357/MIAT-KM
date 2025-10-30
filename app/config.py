# 全域配置文件
# Global configuration for MIAT-KM RAG System

# LLM 模型配置
# LLM Model Configuration
OLLAMA_MODEL = "gpt-oss:20b"  # 更新為新下載的模型
OLLAMA_BASE_URL = "http://ollama:11434"

# Neo4j 配置
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# 模型參數
MODEL_TEMPERATURE = 0.7
MODEL_NUM_PREDICT = -1  # -1表示無限制，讓模型決定
MODEL_TOP_P = 0.9
MODEL_TOP_K = 50

# 資料目錄配置
PDF_DIR = "/app/data/pdf"
MARKDOWN_DIR = "/app/data/markdown"
PROCESSED_DIR = "/app/data/processed"
VECTOR_DB_DIR = "/app/data/vector_db"

# Vector Embedding 配置
# Vector Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"  # BGE-M3 Embedding模型 (多語言、輕量)
EMBEDDING_DEVICE = "cpu"  # 強制使用CPU避免GPU記憶體不足
EMBEDDING_BATCH_SIZE = 16  # BGE-M3較小可以增加批次大小
EMBEDDING_MAX_LENGTH = 512  # 最大序列長度
EMBEDDING_CACHE_DIR = "/root/.cache/huggingface"  # Hugging Face cache目錄

# ChromaDB 配置
# ChromaDB Configuration  
CHROMA_DB_PATH = "/app/data/chroma_db"
CHROMA_COLLECTION_NAME = "miat_documents"

# Document Chunking 配置
# Document Chunking Configuration
CHUNK_SIZE = 512  # 每個chunk的最大字符數
CHUNK_OVERLAP = 50  # chunk之間的重疊字符數
MIN_CHUNK_SIZE = 100  # 最小chunk大小

# ============================================================================
# Prompt Templates 配置
# Prompt Templates Configuration
# ============================================================================

# 三元組提取 Prompt - Gemma 模型 (XML格式)
# Triplet Extraction Prompt for Gemma Model (XML Format)
TRIPLET_EXTRACTION_PROMPT_GEMMA = """從句子中抽取三元組，格式：<三元組>主語|謂語|賓語</三元組>

規則：
1. 只抽取句子中明確存在的關係
2. 主語和賓語必須是具體實體或概念
3. 謂語是動詞或關係詞
4. 每個三元組用 <三元組></三元組> 包圍
5. 如果沒有明確關係，輸出：<三元組>無</三元組>

例子：
句子：張三使用Python開發網站
輸出：
<三元組>張三|使用|Python</三元組>
<三元組>張三|開發|網站</三元組>

句子：GRAFCET是一種控制系統設計方法
輸出：
<三元組>GRAFCET|是|控制系統設計方法</三元組>

現在處理："""

# 三元組提取 Prompt - GPT-OSS 模型 (JSON格式)
# Triplet Extraction Prompt for GPT-OSS Model (JSON Format)
TRIPLET_EXTRACTION_PROMPT_GPT_OSS = """你是一個專業的知識抽取助手。請從給定的文本中抽取知識三元組。

任務：
1. 仔細閱讀文本
2. 識別出重要的實體（人物、地點、概念、方法等）
3. 識別實體之間的關係
4. 以 (主體, 關係, 客體) 的格式輸出三元組

要求：
- 主體和客體應該是具體的實體或概念
- 關係應該清楚表達兩者之間的聯繫
- 避免過於抽象或模糊的關係
- 確保三元組在語義上是正確的

請以以下JSON格式回應：
[
    {"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"},
    {"subject": "主體名稱", "predicate": "關係描述", "object": "客體名稱"}
]

只回應JSON格式，不要其他說明文字。

現在處理文本："""

# Cypher 查詢生成 Prompt
# Cypher Query Generation Prompt
CYPHER_GENERATION_PROMPT = """Task: 根據用戶問題生成Cypher查詢語句

Schema:
- 節點: Entity (屬性: name)
- 關係: RELATION (屬性: name, source)

Rules:
1. 只返回Cypher查詢語句，不要其他說明文字
2. 使用 CONTAINS 和 toLower() 進行模糊匹配
3. 提取用戶問題中的關鍵詞進行查詢
4. 限制返回結果數量 LIMIT 20

Query Templates:
- 單一關鍵詞: MATCH (s:Entity)-[r:RELATION]->(o:Entity) WHERE toLower(s.name) CONTAINS toLower("keyword") OR toLower(o.name) CONTAINS toLower("keyword") RETURN s.name as subject, r.name as predicate, o.name as object LIMIT 20
- 多關鍵詞: MATCH (s:Entity)-[r:RELATION]->(o:Entity) WHERE toLower(s.name) CONTAINS toLower("keyword1") OR toLower(o.name) CONTAINS toLower("keyword1") RETURN s.name as subject, r.name as predicate, o.name as object LIMIT 20

Question: {question}

Cypher:"""

# 知識圖譜問答 Prompt
# Knowledge Graph QA Prompt
KNOWLEDGE_GRAPH_QA_PROMPT = """你是一個專業的知識問答助手。請基於以下從知識圖譜檢索到的信息來詳細回答用戶的問題。

從知識圖譜檢索到的相關信息：
{context}

用戶問題：{question}

請根據上述檢索到的知識信息，提供一個詳細、完整且有條理的回答。要求：

1. **完整性**：盡可能整合所有相關的檢索信息
2. **詳細性**：提供豐富的細節和背景信息
3. **結構化**：使用清晰的段落和邏輯結構
4. **準確性**：嚴格基於檢索到的知識，不要添加不存在的信息
5. **關聯性**：解釋不同信息之間的關係和聯繫

如果檢索到的信息不足以完全回答問題，請明確說明哪些方面的信息不足，並基於已有信息提供盡可能詳細的回答。

請開始你的詳細回答：
"""

# RAG Chain-of-Thought 問答 Prompt
# RAG Chain-of-Thought QA Prompt
RAG_COT_PROMPT = """你是一個知識問答助手。請根據以下提供的知識上下文來回答用戶的問題。

知識上下文：
{knowledge_context}

用戶問題：{user_query}

請按照以下格式回答，先思考再給出最終答案：

<thinking>
[請在這裡寫出你的思考過程：
1. 分析用戶問題的關鍵點
2. 從知識上下文中找出相關信息
3. 進行邏輯推理和分析
4. 組織答案結構]
</thinking>

<answer>
[請在這裡給出最終的詳細答案，基於上述思考過程和知識上下文]
</answer>

請確保思考過程詳細清晰，最終答案準確完整。"""