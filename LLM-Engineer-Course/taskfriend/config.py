# Configuration settings for TaskFriend

# Core LLM settings
MODEL = "qwen-plus"
TEMPERATURE = 0.7
TOP_P = 0.8
CONTEXT_WINDOW = 2000  # Increased to better handle conversation history
TOKEN_FACTOR = 4  # Estimate: ~4 characters per token

# Context management settings
USE_CONTEXT_WINDOW = False
SHOW_TRUNCATED = False
SUMMARIZE_DROPPED = False
SHOW_CONTEXT_PREVIEW = False
SHOW_SOURCES = True

# RAG settings
MODEL_EMBEDDING_NAME = "text-embedding-v3"
PERSIST_PATH = "knowledge_base/taskfriend"
DOCUMENT_PATH = "./docs"
DASHSCOPE_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Logging settings
LOGGING_LEVEL = "ERROR"
