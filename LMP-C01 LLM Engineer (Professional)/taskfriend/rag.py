from pathlib import Path
from typing import Optional
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
import os
import logging
import dashscope

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config
from .rag_config import (
    MODEL_EMBEDDING_NAME, MODEL_LLM_NAME,
    PERSIST_PATH, DOCUMENT_PATH, DASHSCOPE_API_BASE,
    LOGGING_LEVEL, DASHSCOPE_BASE_HTTP
)

# Set global DashScope API URL (domestic or international based on config)
dashscope.base_http_api_url = DASHSCOPE_BASE_HTTP

def get_llm():
    """Get configured LLM instance"""
    from .config import DASHSCOPE_API_KEY

    return OpenAILike(
        model=MODEL_LLM_NAME,
        api_base=DASHSCOPE_API_BASE,
        api_key=DASHSCOPE_API_KEY,
        is_chat_model=True
    )

def get_embedding_model():
    """Get configured embedding model"""
    return DashScopeEmbedding(model_name=MODEL_EMBEDDING_NAME)

def load_documents(document_path: str = DOCUMENT_PATH):
    """Load documents from specified path"""
    logger.info(f"Loading documents from {document_path}")
    documents = SimpleDirectoryReader(document_path).load_data()
    logger.info(f"Loaded {len(documents)} documents")
    return documents

def create_index(document_path: str = DOCUMENT_PATH):
    """Create index from documents"""
    logger.info(f"Creating index from documents in {document_path}")
    documents = load_documents(document_path)
    embed_model = get_embedding_model()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    logger.info("Index created successfully")
    return index

def reindex(document_path: str = DOCUMENT_PATH, persist_path: str = PERSIST_PATH):
    """Build and persist the index"""
    logger.info(f"Building index from documents in {document_path}")
    index = create_index(document_path)

    logger.info(f"Saving index to {persist_path}")
    index.storage_context.persist(persist_dir=persist_path)
    logger.info(f"Index saved to {persist_path}")
    print(f"✅ Index rebuilt and saved to `{persist_path}`")
    return index

def load_index(persist_path: str = PERSIST_PATH):
    """Load a previously saved index"""
    logger.info(f"Loading index from {persist_path}")
    storage_context = StorageContext.from_defaults(persist_dir=persist_path)
    embed_model = get_embedding_model()
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    logger.info("Index loaded successfully")
    print(f"✅ Index loaded from `{persist_path}`")
    return index

def query_engine(index, streaming: bool = True):
    """Create a query engine from the index"""
    logger.info("Creating query engine")
    llm = get_llm()

    query_engine = index.as_query_engine(
        streaming=streaming,
        llm=llm,
        similarity_top_k=2
    )
    logger.info("Query engine created successfully")
    return query_engine
