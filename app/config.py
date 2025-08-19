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
MODEL_NUM_PREDICT = 512
MODEL_TOP_P = 0.9
MODEL_TOP_K = 50

# 資料目錄配置
PDF_DIR = "/app/data/pdf"
MARKDOWN_DIR = "/app/data/markdown"
PROCESSED_DIR = "/app/data/processed"
VECTOR_DB_DIR = "/app/data/vector_db"

# Vector Embedding 配置
# Vector Embedding Configuration
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"  # QWEN3 Embedding模型
EMBEDDING_DEVICE = "cpu"  # 強制使用CPU避免GPU記憶體不足
EMBEDDING_BATCH_SIZE = 8   # 減小批次大小避免記憶體問題
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