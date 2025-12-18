"""
Configuration file for RAG Chatbot with ML-enhanced performance
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class ModelConfig:
    """Configuration for different AI models"""

    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40


@dataclass
class VectorStoreConfig:
    """Configuration for vector databases"""

    default_store: str = "chromadb"  # chromadb, faiss, pinecone, weaviate
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.3
    max_retrieval_docs: int = 10


@dataclass
class MLConfig:
    """Configuration for ML algorithms"""

    enable_reranking: bool = True
    enable_query_expansion: bool = True
    enable_semantic_caching: bool = True
    enable_performance_monitoring: bool = True

    # ML model configurations
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_expansion_model: str = "t5-base"
    classification_model: str = "distilbert-base-uncased"

    # Performance thresholds
    response_time_threshold: float = 5.0  # seconds
    memory_threshold: float = 80.0  # percentage
    cache_ttl: int = 3600  # seconds


@dataclass
class StreamlitConfig:
    """Configuration for Streamlit app"""

    page_title: str = "Advanced RAG Chatbot"
    page_icon: str = "🤖"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    theme: str = "light"

    # UI configurations
    max_file_size: int = 200  # MB
    supported_formats: List[str] = None
    enable_authentication: bool = True
    session_timeout: int = 3600  # seconds

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                ".pdf",
                ".txt",
                ".docx",
                ".csv",
                ".json",
                ".md",
                ".html",
            ]


@dataclass
class DatabaseConfig:
    """Configuration for databases"""

    # Vector databases
    chromadb_path: str = "./data/chromadb"
    faiss_index_path: str = "./data/faiss_index"

    # Traditional databases
    sqlite_path: str = "./data/app.db"
    redis_url: str = "redis://localhost:6379"
    mongodb_url: str = "mongodb://localhost:27017"

    # Cloud databases (optional)
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    weaviate_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None


@dataclass
class SecurityConfig:
    """Security configuration"""

    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    api_key_encryption: bool = True

    # JWT configuration
    jwt_secret_key: str = "your-secret-key-here"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24


class Config:
    """Main configuration class"""

    def __init__(self):
        # Load environment variables
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        # Initialize configurations
        self.model = ModelConfig()
        self.vector_store = VectorStoreConfig()
        self.ml = MLConfig()
        self.streamlit = StreamlitConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()

        # Application settings
        self.app_name = "Advanced RAG Chatbot"
        self.version = "2.0.0"
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Performance monitoring
        self.monitoring = {
            "enable_metrics": True,
            "enable_logging": True,
            "log_level": "INFO",
            "metrics_port": 8080,
            "health_check_endpoint": "/health",
        }

        # ML experiment tracking
        self.experiment_tracking = {
            "enable_mlflow": True,
            "enable_wandb": False,
            "mlflow_tracking_uri": "./mlruns",
            "experiment_name": "rag_chatbot_experiments",
        }

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY environment variable is required")

        if self.model.chunk_size <= 0:
            errors.append("Chunk size must be positive")

        if self.model.chunk_overlap >= self.model.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")

        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False

        return True

    def get_model_config(self, model_type: str) -> Dict:
        """Get configuration for specific model type"""
        configs = {
            "gemini": {
                "api_key": self.gemini_api_key,
                "model": self.model.gemini_model,
                "max_tokens": self.model.max_tokens,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "top_k": self.model.top_k,
            },
            "openai": {
                "api_key": self.openai_api_key,
                "model": "gpt-3.5-turbo",
                "max_tokens": self.model.max_tokens,
                "temperature": self.model.temperature,
            },
            "cohere": {
                "api_key": self.cohere_api_key,
                "model": "command",
                "max_tokens": self.model.max_tokens,
                "temperature": self.model.temperature,
            },
        }

        return configs.get(model_type, {})

    def get_vector_store_config(self, store_type: str) -> Dict:
        """Get configuration for specific vector store"""
        configs = {
            "chromadb": {
                "persist_directory": self.database.chromadb_path,
                "collection_name": "rag_documents",
            },
            "faiss": {
                "index_path": self.database.faiss_index_path,
                "dimension": self.model.embedding_dimension,
            },
            "pinecone": {
                "api_key": self.database.pinecone_api_key,
                "environment": self.database.pinecone_environment,
                "index_name": "rag-chatbot",
            },
            "weaviate": {"url": self.database.weaviate_url, "class_name": "Document"},
        }

        return configs.get(store_type, {})


# Global configuration instance
config = Config()

# Performance monitoring settings
PERFORMANCE_METRICS = {
    "response_time": [],
    "memory_usage": [],
    "cpu_usage": [],
    "query_count": 0,
    "error_count": 0,
    "cache_hit_rate": 0.0,
}

# ML model paths
ML_MODEL_PATHS = {
    "query_classifier": "./models/query_classifier",
    "intent_classifier": "./models/intent_classifier",
    "sentiment_analyzer": "./models/sentiment_analyzer",
    "topic_extractor": "./models/topic_extractor",
    "quality_scorer": "./models/quality_scorer",
}

# Supported file formats for document ingestion
SUPPORTED_FORMATS = {
    ".pdf": "PDF documents",
    ".txt": "Text files",
    ".docx": "Word documents",
    ".md": "Markdown files",
    ".html": "HTML files",
    ".csv": "CSV files",
    ".json": "JSON files",
    ".xml": "XML files",
}

# Default system prompts
SYSTEM_PROMPTS = {
    "default": """You are an advanced AI assistant with access to a knowledge base.
    Provide accurate, helpful, and contextual responses based on the retrieved information.
    If you don't know something, say so clearly.""",
    "technical": """You are a technical expert AI assistant. Provide detailed,
    accurate technical information with examples and best practices when relevant.""",
    "creative": """You are a creative AI assistant. Provide imaginative,
    engaging responses while maintaining accuracy based on the available information.""",
    "analytical": """You are an analytical AI assistant. Provide data-driven,
    logical responses with clear reasoning and evidence from the knowledge base.""",
}
