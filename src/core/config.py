"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class ModelConfig(BaseSettings):
    """LLM and embedding model configuration."""

    # LLM Configuration
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_max_tokens: int = 2048
    llm_context_window: int = 8192

    # Embedding Configuration
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768
    embed_batch_size: int = 32
    embed_normalize: bool = True

    # Reranker Configuration
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512

    model_config = SettingsConfigDict(env_prefix="")


class RetrievalConfig(BaseSettings):
    """Retrieval pipeline configuration."""

    # Stage 1: Vector Search
    vector_top_k: int = 100
    vector_distance_metric: str = "cosine"
    vector_ef_search: int = 128

    # Stage 2: Reranking
    rerank_enabled: bool = True
    rerank_top_n: int = 25
    rerank_score_threshold: float = 0.3

    # Stage 3: Graph Expansion
    graph_expansion_enabled: bool = True
    graph_max_hops: int = 1
    graph_max_neighbors: int = 10
    graph_similarity_threshold: float = 0.75
    graph_include_sequential: bool = True

    # Stage 4: Context Assembly
    max_context_tokens: int = 8192
    overlap_handling: str = "merge"
    sort_by: List[str] = ["relevance", "document", "page"]
    include_metadata: bool = True

    model_config = SettingsConfigDict(env_prefix="")


class ChunkingConfig(BaseSettings):
    """Chunking strategy configuration."""

    strategy: str = "hybrid"
    min_chunk_size: int = 40
    max_chunk_size: int = 300
    overlap: float = 0.2

    # Special handling
    preserve_tables: bool = True
    preserve_lists: bool = True
    preserve_code_blocks: bool = True
    keep_headers_with_content: bool = True

    # Metadata
    include_page_numbers: bool = True
    include_section_titles: bool = True
    include_document_metadata: bool = True
    include_position: bool = True

    model_config = SettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    """Main application settings."""

    # Project Paths
    # Self-locate the repo root from this file's location so the app works
    # for any collaborator regardless of where they clone it, rather than
    # requiring PROJECT_ROOT to be hand-edited per machine.
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    source_docs_path: Path = Field(default=Path("soc"))
    data_path: Path = Field(default=Path("data"))
    vector_store_path: Path = Field(default=Path("data/vector_store"))
    graph_db_path: Path = Field(default=Path("data/graph_db"))
    metadata_db_path: Path = Field(default=Path("data/metadata.db"))

    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Web UI Configuration
    web_ui_port: int = 7860

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/rag.log"

    # Auth & Security
    # Required for encrypting stored OpenAI/Anthropic API keys (Fernet) and
    # signing session cookies. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    secret_key: str = ""
    session_cookie_name: str = "rag_session"
    session_ttl_days: int = 30
    # Only send the session cookie over HTTPS. Keep False for local/plain-HTTP
    # dev; set True once served behind TLS.
    cookie_secure: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
    )

    @property
    def full_source_docs_path(self) -> Path:
        """Get absolute path to source documents."""
        return self.project_root / self.source_docs_path

    @property
    def full_data_path(self) -> Path:
        """Get absolute path to data directory."""
        return self.project_root / self.data_path

    @property
    def full_vector_store_path(self) -> Path:
        """Get absolute path to vector store."""
        return self.project_root / self.vector_store_path

    @property
    def full_graph_db_path(self) -> Path:
        """Get absolute path to graph database."""
        return self.project_root / self.graph_db_path

    @property
    def full_metadata_db_path(self) -> Path:
        """Get absolute path to metadata database."""
        return self.project_root / self.metadata_db_path


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _default_config_dir() -> Path:
    """Repo-root-relative config/ dir, independent of the process's cwd."""
    return settings.project_root / "config"


def load_model_config(config_dir: Optional[Path] = None) -> ModelConfig:
    """Load model configuration from YAML."""
    config_dir = config_dir or _default_config_dir()
    config_path = config_dir / "models.yaml"
    if config_path.exists():
        config_data = load_yaml_config(config_path)
        return ModelConfig(
            llm_provider=config_data.get("llm", {}).get("provider", "ollama"),
            llm_model=config_data.get("llm", {}).get("model", "llama3.1:8b"),
            llm_temperature=config_data.get("llm", {}).get("temperature", 0.1),
            llm_top_p=config_data.get("llm", {}).get("top_p", 0.9),
            llm_max_tokens=config_data.get("llm", {}).get("max_tokens", 2048),
            llm_context_window=config_data.get("llm", {}).get("context_window", 8192),
            embed_provider=config_data.get("embedding", {}).get("provider", "ollama"),
            embed_model=config_data.get("embedding", {}).get("model", "nomic-embed-text"),
            embed_dimensions=config_data.get("embedding", {}).get("dimensions", 768),
            embed_batch_size=config_data.get("embedding", {}).get("batch_size", 32),
            embed_normalize=config_data.get("embedding", {}).get("normalize", True),
            reranker_model=config_data.get("reranker", {}).get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            reranker_batch_size=config_data.get("reranker", {}).get("batch_size", 16),
            reranker_max_length=config_data.get("reranker", {}).get("max_length", 512),
        )
    return ModelConfig()


def load_retrieval_config(config_dir: Optional[Path] = None) -> RetrievalConfig:
    """Load retrieval pipeline configuration from YAML."""
    config_dir = config_dir or _default_config_dir()
    config_path = config_dir / "retrieval.yaml"
    if config_path.exists():
        data = load_yaml_config(config_path)
        stage1 = data.get("stage1_vector_search", {})
        stage2 = data.get("stage2_reranking", {})
        stage3 = data.get("stage3_graph_expansion", {})
        stage4 = data.get("stage4_context_assembly", {})
        return RetrievalConfig(
            vector_top_k=stage1.get("top_k", 100),
            vector_distance_metric=stage1.get("distance_metric", "cosine"),
            vector_ef_search=stage1.get("ef_search", 128),
            rerank_enabled=stage2.get("enabled", True),
            rerank_top_n=stage2.get("top_n", 25),
            rerank_score_threshold=stage2.get("score_threshold", 0.3),
            graph_expansion_enabled=stage3.get("enabled", False),
            graph_max_hops=stage3.get("max_hops", 1),
            graph_max_neighbors=stage3.get("max_neighbors", 10),
            graph_similarity_threshold=stage3.get("similarity_threshold", 0.75),
            graph_include_sequential=stage3.get("include_sequential", True),
            max_context_tokens=stage4.get("max_context_tokens", 8192),
            overlap_handling=stage4.get("overlap_handling", "merge"),
            sort_by=stage4.get("sort_by", ["relevance", "document", "page"]),
            include_metadata=stage4.get("include_metadata", True),
        )
    return RetrievalConfig()


def load_chunking_config(config_dir: Optional[Path] = None) -> ChunkingConfig:
    """Load chunking strategy configuration from YAML."""
    config_dir = config_dir or _default_config_dir()
    config_path = config_dir / "chunking.yaml"
    if config_path.exists():
        data = load_yaml_config(config_path)
        semantic = data.get("semantic_chunking", {})
        hybrid = data.get("hybrid", {})
        special = data.get("special_handling", {})
        meta = data.get("metadata", {})
        return ChunkingConfig(
            strategy=data.get("strategy", "hybrid"),
            min_chunk_size=semantic.get("min_chunk_size", 40),
            max_chunk_size=hybrid.get("max_chunk_size", 300),
            overlap=hybrid.get("overlap", 0.2),
            preserve_tables=special.get("preserve_tables", True),
            preserve_lists=special.get("preserve_lists", True),
            preserve_code_blocks=special.get("preserve_code_blocks", True),
            keep_headers_with_content=special.get("keep_headers_with_content", True),
            include_page_numbers=meta.get("include_page_numbers", True),
            include_section_titles=meta.get("include_section_titles", True),
            include_document_metadata=meta.get("include_document_metadata", True),
            include_position=meta.get("include_position", True),
        )
    return ChunkingConfig()


# Global settings instance
settings = Settings()
