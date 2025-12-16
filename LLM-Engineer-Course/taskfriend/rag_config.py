# Configuration for RAG functionality
import os

# Model configuration
MODEL_EMBEDDING_NAME = "text-embedding-v3"
MODEL_LLM_NAME = "qwen-plus"

# File paths
PERSIST_PATH = "knowledge_base/taskfriend"
DOCUMENT_PATH = "./docs"

# API configuration
_region = os.getenv("DASHSCOPE_REGION", "intl").strip().lower()

if _region in {"cn", "china", "domestic"}:
	DASHSCOPE_BASE_HTTP = "https://dashscope.aliyuncs.com/api/v1"
	DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
else:
	DASHSCOPE_BASE_HTTP = "https://dashscope-intl.aliyuncs.com/api/v1"
	DASHSCOPE_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Logging
LOGGING_LEVEL = "ERROR"
