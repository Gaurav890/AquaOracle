"""Ollama embedding client."""

from typing import List, Optional
import ollama
from loguru import logger
from tqdm import tqdm


class OllamaEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        batch_size: int = 32,
    ):
        """
        Initialize Ollama embedder.

        Args:
            model: Embedding model name
            host: Ollama server host
            batch_size: Batch size for embedding generation
        """
        self.logger = logger.bind(name="OllamaEmbedder")
        self.model = model
        self.host = host
        self.batch_size = batch_size

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            # Try to generate a test embedding
            response = ollama.embeddings(
                model=self.model,
                prompt="test"
            )
            self.logger.info(f"Connected to Ollama server with model {self.model}")
            self.logger.info(f"Embedding dimensions: {len(response['embedding'])}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            return response['embedding']

        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        embeddings = []

        iterator = tqdm(texts, desc="Generating embeddings") if show_progress else texts

        for text in iterator:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)

            except Exception as e:
                self.logger.warning(f"Failed to embed text (using zero vector): {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)  # Default dimension

        return embeddings

    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents with batching.

        Args:
            texts: List of texts to embed
            batch_size: Batch size (uses default if None)
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        total_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc="Embedding batches")

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_batch(batch, show_progress=False)
            all_embeddings.extend(batch_embeddings)

        self.logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        test_embedding = self.embed_text("test")
        return len(test_embedding)
