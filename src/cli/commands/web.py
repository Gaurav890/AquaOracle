"""Web UI CLI command."""

import click
import uvicorn
from rich.console import Console

from src.core.config import settings

console = Console()


@click.command()
@click.option('--port', default=None, type=int, help='Port to serve on (default: from settings)')
def web(port):
    """
    Launch the web UI — chat-based Q&A plus document management.

    Example:
        rag web
        rag web --port 8080
    """
    resolved_port = port or settings.web_ui_port
    console.print(f"\n[bold cyan]Starting web UI[/bold cyan] at [bold]http://127.0.0.1:{resolved_port}[/bold]\n")

    uvicorn.run("src.api.main:create_app", factory=True, host="127.0.0.1", port=resolved_port)
