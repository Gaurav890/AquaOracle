"""Custom exceptions for RAG platform."""


class RAGException(Exception):
    """Base exception for RAG platform."""
    pass


class ConfigurationError(RAGException):
    """Configuration-related errors."""
    pass


class DocumentProcessingError(RAGException):
    """Document processing errors."""
    pass


class EmbeddingError(RAGException):
    """Embedding generation errors."""
    pass


class RetrievalError(RAGException):
    """Retrieval pipeline errors."""
    pass


class GenerationError(RAGException):
    """LLM generation errors."""
    pass


class StorageError(RAGException):
    """Storage and database errors."""
    pass


class ValidationError(RAGException):
    """Input validation errors."""
    pass
