"""Index management CLI commands."""

import click
from rich.console import Console
from rich.table import Table

from src.core.config import settings
from src.indexing.metadata_store import MetadataStore
from src.indexing.vector_store import VectorStore


console = Console()


@click.group()
def index():
    """Manage document indices."""
    pass


@index.command()
def status():
    """Show index statistics."""
    try:
        metadata_store = MetadataStore(settings.full_metadata_db_path)
        vector_store = VectorStore(path=settings.full_vector_store_path)

        # Get stats
        metadata_stats = metadata_store.get_stats()
        vector_stats = vector_store.get_stats()

        # Create table
        table = Table(title="Index Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")

        table.add_row("Documents", str(metadata_stats.get("document_count", 0)))
        table.add_row("Total Pages", str(metadata_stats.get("total_pages", 0)))
        table.add_row("Chunks", str(metadata_stats.get("chunk_count", 0)))
        table.add_row("Tables", str(metadata_stats.get("table_count", 0)))
        table.add_row("Figures", str(metadata_stats.get("figure_count", 0)))
        table.add_row("Vector Count", str(vector_stats.get("total_vectors", 0)))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting index status: {e}[/red]")


@index.command()
def list():
    """List all indexed documents."""
    try:
        metadata_store = MetadataStore(settings.full_metadata_db_path)
        documents = metadata_store.list_documents()

        if not documents:
            console.print("[yellow]No documents indexed yet.[/yellow]")
            return

        table = Table(title=f"Indexed Documents ({len(documents)})")
        table.add_column("Document", style="cyan")
        table.add_column("Pages", style="green")
        table.add_column("Chunks", style="green")
        table.add_column("Organization", style="yellow")
        table.add_column("Year", style="yellow")

        for doc in documents:
            table.add_row(
                doc.get("title", doc.get("file_name", "Unknown"))[:50],
                str(doc.get("page_count", 0)),
                str(doc.get("chunk_count", 0)),
                doc.get("organization", "N/A"),
                str(doc.get("year", "N/A")),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing documents: {e}[/red]")
