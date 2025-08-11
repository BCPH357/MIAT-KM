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