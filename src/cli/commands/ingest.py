"""Document ingestion CLI commands."""

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
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


console = Console()
model_config = load_model_config()
chunking_config = load_chunking_config()


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--rebuild', is_flag=True, help='Rebuild index from scratch')
def ingest(path, rebuild):
    """
    Ingest documents from PATH (file or directory).

    Examples:
        rag ingest soc/
        rag ingest "soc/CDC, 2017.pdf"
    """
    path = Path(path)

    if rebuild:
        console.print("[yellow]Rebuilding index from scratch...[/yellow]")
        metadata_store = MetadataStore(settings.full_metadata_db_path)
        vector_store = VectorStore(
            path=settings.full_vector_store_path,
            embedding_dim=model_config.embed_dimensions,
        )
        metadata_store.clear_all()
        vector_store.clear()
        # Qdrant's embedded mode holds an exclusive lock on the storage path;
        # release it now so ingest_single_pdf can open its own VectorStore below.
        vector_store.close()
        console.print("[yellow]Cleared existing indices.[/yellow]")

    if path.is_file() and path.suffix.lower() == '.pdf':
        ingest_single_pdf(path)
    elif path.is_dir():
        ingest_directory(path)
    else:
        console.print(f"[red]Error: {path} is not a PDF file or directory[/red]")
        raise click.Abort()


def ingest_single_pdf(pdf_path: Path):
    """Ingest a single PDF file."""
    console.print(f"\n[bold]Ingesting:[/bold] {pdf_path.name}")

    try:
        # Initialize components
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:

            # Step 1: Parse PDF
            task1 = progress.add_task("[cyan]Parsing PDF...", total=1)
            pdf_doc = pdf_parser.parse(pdf_path)
            progress.update(task1, completed=1)

            # Step 2: Extract tables
            task2 = progress.add_task("[cyan]Extracting tables...", total=1)
            tables = table_extractor.extract_tables(pdf_path)
            progress.update(task2, completed=1)

            # Step 3: Extract figures
            task3 = progress.add_task("[cyan]Extracting figures...", total=1)
            figures = figure_extractor.extract_figures(pdf_path)
            progress.update(task3, completed=1)

            # Step 4: Extract metadata
            task4 = progress.add_task("[cyan]Extracting metadata...", total=1)
            doc_metadata = metadata_extractor.extract(pdf_doc.metadata, pdf_path)
            doc_id = metadata_extractor.create_doc_id(pdf_path)
            progress.update(task4, completed=1)

            # Step 5: Chunk document
            task5 = progress.add_task("[cyan]Chunking document...", total=1)
            pages = [(page.page_number, page.text) for page in pdf_doc.pages]
            chunks = chunker.chunk_document(
                pages=pages,
                doc_id=doc_id,
                metadata=doc_metadata,
            )
            progress.update(task5, completed=1)

            # Step 6: Generate embeddings
            task6 = progress.add_task(
                f"[cyan]Generating embeddings for {len(chunks)} chunks...",
                total=len(chunks)
            )
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = embedder.embed_documents(chunk_texts, show_progress=False)
            failed_count = sum(1 for e in embeddings if e is None)

            for _ in range(len(chunks)):
                progress.update(task6, advance=1)

            # Step 7: Store in databases
            task7 = progress.add_task("[cyan]Storing in databases...", total=5)

            # Store document metadata
            doc_metadata["chunk_count"] = len(chunks)
            doc_metadata["table_count"] = len(tables)
            doc_metadata["figure_count"] = len(figures)
            metadata_store.add_document(doc_id, doc_metadata)
            progress.update(task7, advance=1)

            # Store chunks in metadata DB
            for chunk in chunks:
                chunk_id = f"{doc_id}_chunk_{chunk.chunk_index}"
                chunk_data = {
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "page_numbers": chunk.page_numbers,
                    "line_ranges": chunk.line_ranges,
                    "section_title": chunk.section_title,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "metadata": chunk.metadata,
                }
                metadata_store.add_chunk(chunk_id, doc_id, chunk_data)
            progress.update(task7, advance=1)

            # Store extracted tables and figures (re-ingesting the same doc
            # shouldn't accumulate duplicate rows, so clear first)
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
            progress.update(task7, advance=1)

            # Store embeddings in vector DB — skip chunks whose embedding
            # failed rather than storing a placeholder that would silently
            # sit in the index as unsearchable.
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
            progress.update(task7, advance=1)
            if valid:
                valid_ids, valid_embeddings, valid_payloads = map(list, zip(*valid))
                vector_store.add_embeddings(valid_ids, valid_embeddings, valid_payloads)
            progress.update(task7, advance=1)

        # Summary
        console.print(f"\n[bold green]✓ Successfully ingested {pdf_path.name}[/bold green]")
        console.print(f"  • Pages: {pdf_doc.page_count}")
        console.print(f"  • Chunks: {len(chunks)}")
        console.print(f"  • Tables: {len(tables)}")
        console.print(f"  • Figures: {len(figures)}")
        if failed_count:
            console.print(
                f"  [yellow]⚠ {failed_count} chunk(s) failed to embed and are "
                f"not searchable via vector search[/yellow]"
            )

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to ingest {pdf_path.name}[/bold red]")
        console.print(f"  Error: {e}")
        logger.exception("Ingestion failed")
        raise click.Abort()


def ingest_directory(dir_path: Path):
    """Ingest all PDFs in a directory."""
    pdf_files = list(dir_path.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {dir_path}[/yellow]")
        return

    console.print(f"\n[bold]Found {len(pdf_files)} PDF files[/bold]\n")

    for i, pdf_file in enumerate(pdf_files, 1):
        console.print(f"[bold cyan]({i}/{len(pdf_files)})[/bold cyan]")
        try:
            ingest_single_pdf(pdf_file)
        except Exception:
            console.print(f"[red]Skipping {pdf_file.name} due to error[/red]")
            continue

    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
