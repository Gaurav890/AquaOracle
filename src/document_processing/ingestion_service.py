"""Shared document ingestion pipeline.

Both the CLI (`rag ingest`) and the web app's upload panel need to parse,
chunk, embed, and store a PDF the same way — this is the one place that
logic lives, so the two consumers can't drift apart on it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from src.core.config import settings, load_model_config, load_chunking_config
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.table_extractor import TableExtractor
from src.document_processing.figure_extractor import FigureExtractor
from src.document_processing.chunker import Chunker
from src.document_processing.metadata_extractor import MetadataExtractor
from src.embedding.ollama_embedder import OllamaEmbedder
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_store import VectorStore


@dataclass
class IngestionResult:
    """Outcome of ingesting one PDF."""

    doc_id: str
    file_name: str
    success: bool
    pages: int = 0
    chunks: int = 0
    tables: int = 0
    figures: int = 0
    failed_embeddings: int = 0
    error: Optional[str] = None


# Called with a short human-readable stage description ("Parsing PDF...").
ProgressCallback = Callable[[str], None]


def ingest_pdf(
    pdf_path: Path,
    on_progress: Optional[ProgressCallback] = None,
    owner_user_id: Optional[str] = None,
    is_shared: bool = False,
) -> IngestionResult:
    """
    Parse, chunk, embed, and store a single PDF.

    Args:
        pdf_path: Path to the PDF file
        on_progress: Optional callback invoked with a stage description as
            ingestion proceeds (e.g. for a progress bar or status line)
        owner_user_id: Web-upload caller's user id. None (the default)
            preserves today's CLI behavior — the document is visible to
            everyone, the same way rag ingest already works, since the CLI
            has no concept of "which user" is running it.
        is_shared: Whether this document should be visible from every one
            of the owning user's chats, not just the chat it was uploaded
            into. Ignored when owner_user_id is None.

    Returns:
        IngestionResult describing what happened, success or failure —
        never raises, so callers (CLI, web UI) don't each need their own
        try/except around every ingestion.
    """
    log = logger.bind(name="IngestionService")
    vector_store = None

    def report(message: str) -> None:
        if on_progress:
            on_progress(message)

    try:
        model_config = load_model_config()
        chunking_config = load_chunking_config()

        pdf_parser = PDFParser()
        table_extractor = TableExtractor()
        figure_extractor = FigureExtractor()
        chunker = Chunker(
            max_chunk_size=chunking_config.max_chunk_size,
            min_chunk_size=chunking_config.min_chunk_size,
            overlap=chunking_config.overlap,
        )
        metadata_extractor = MetadataExtractor()
        embedder = OllamaEmbedder(
            model=settings.ollama_embed_model,
            host=settings.ollama_host,
            batch_size=model_config.embed_batch_size,
        )
        metadata_store = MetadataStore(settings.full_metadata_db_path)
        vector_store = VectorStore(
            path=settings.full_vector_store_path,
            embedding_dim=model_config.embed_dimensions,
        )

        report("Parsing PDF...")
        pdf_doc = pdf_parser.parse(pdf_path)

        report("Extracting tables...")
        tables = table_extractor.extract_tables(pdf_path)

        report("Extracting figures...")
        figures = figure_extractor.extract_figures(pdf_path)

        report("Extracting metadata...")
        doc_metadata = metadata_extractor.extract(pdf_doc.metadata, pdf_path)
        doc_metadata["owner_user_id"] = owner_user_id
        doc_metadata["is_shared"] = is_shared
        doc_id = metadata_extractor.create_doc_id(pdf_path)

        report("Chunking document...")
        pages = [(page.page_number, page.text) for page in pdf_doc.pages]
        chunks = chunker.chunk_document(pages=pages, doc_id=doc_id, metadata=doc_metadata)

        report(f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(chunk_texts, show_progress=False)
        failed_count = sum(1 for e in embeddings if e is None)

        report("Storing in databases...")
        doc_metadata["chunk_count"] = len(chunks)
        doc_metadata["table_count"] = len(tables)
        doc_metadata["figure_count"] = len(figures)
        metadata_store.add_document(doc_id, doc_metadata)

        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
            metadata_store.add_chunk(chunk_id, doc_id, {
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "page_numbers": chunk.page_numbers,
                "line_ranges": chunk.line_ranges,
                "section_title": chunk.section_title,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "metadata": chunk.metadata,
            })

        # Re-ingesting the same doc shouldn't accumulate duplicate rows.
        metadata_store.clear_tables_and_figures(doc_id)
        for table in tables:
            metadata_store.add_table(doc_id, {
                "page_number": table.page_number,
                "row_count": table.row_count,
                "col_count": table.col_count,
                "has_header": table.has_header,
                "data": table.data,
            })
        for figure in figures:
            metadata_store.add_figure(doc_id, {
                "page_number": figure.page_number,
                "image_index": figure.image_index,
                "caption": figure.caption,
                "width": figure.width,
                "height": figure.height,
            })

        # Skip chunks whose embedding failed rather than storing a
        # placeholder that would silently sit in the index unsearchable.
        chunk_ids = [f"{doc_id}_chunk_{chunk.chunk_index}" for chunk in chunks]
        payloads = [
            {
                "doc_id": doc_id,
                "text": chunk.text,
                "page_numbers": chunk.page_numbers,
                "line_ranges": chunk.line_ranges,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]
        valid = [
            (cid, emb, payload)
            for cid, emb, payload in zip(chunk_ids, embeddings, payloads)
            if emb is not None
        ]
        if valid:
            valid_ids, valid_embeddings, valid_payloads = map(list, zip(*valid))
            vector_store.add_embeddings(valid_ids, valid_embeddings, valid_payloads)

        report("Done")
        return IngestionResult(
            doc_id=doc_id,
            file_name=pdf_path.name,
            success=True,
            pages=pdf_doc.page_count,
            chunks=len(chunks),
            tables=len(tables),
            figures=len(figures),
            failed_embeddings=failed_count,
        )

    except Exception as e:
        log.exception(f"Ingestion failed for {pdf_path.name}")
        return IngestionResult(doc_id="", file_name=pdf_path.name, success=False, error=str(e))

    finally:
        # Qdrant's embedded mode holds an exclusive lock on the storage
        # path for as long as the client is open — release it promptly so
        # a concurrent query (or another ingestion) isn't blocked by it.
        if vector_store is not None:
            vector_store.close()
