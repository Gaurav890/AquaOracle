"""Document ingestion CLI commands."""

import click
from pathlib import Path
from rich.console import Console
from loguru import logger

from src.core.config import settings, load_model_config
from src.document_processing.ingestion_service import ingest_pdf
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_store import VectorStore


console = Console()
model_config = load_model_config()


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
    """Ingest a single PDF file, displaying progress and a summary."""
    console.print(f"\n[bold]Ingesting:[/bold] {pdf_path.name}")

    with console.status("[cyan]Starting...", spinner="dots") as status:
        result = ingest_pdf(pdf_path, on_progress=lambda msg: status.update(f"[cyan]{msg}"))

    if not result.success:
        console.print(f"\n[bold red]✗ Failed to ingest {pdf_path.name}[/bold red]")
        console.print(f"  Error: {result.error}")
        logger.error(f"Ingestion failed for {pdf_path.name}: {result.error}")
        raise click.Abort()

    console.print(f"\n[bold green]✓ Successfully ingested {pdf_path.name}[/bold green]")
    console.print(f"  • Pages: {result.pages}")
    console.print(f"  • Chunks: {result.chunks}")
    console.print(f"  • Tables: {result.tables}")
    console.print(f"  • Figures: {result.figures}")
    if result.failed_embeddings:
        console.print(
            f"  [yellow]⚠ {result.failed_embeddings} chunk(s) failed to embed and are "
            f"not searchable via vector search[/yellow]"
        )


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
        except click.Abort:
            console.print(f"[red]Skipping {pdf_file.name} due to error[/red]")
            continue

    console.print("\n[bold green]✓ Ingestion complete![/bold green]")
