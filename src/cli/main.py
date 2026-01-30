"""Main CLI entry point."""

import click
from pathlib import Path
from src.core.logging_config import setup_logging
from src.core.config import settings


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', default='INFO', help='Set log level')
def cli(verbose, log_level):
    """RAG Platform - Privacy-focused retrieval-augmented generation."""
    if verbose:
        log_level = 'DEBUG'

    setup_logging(log_level=log_level, log_file=settings.log_file)


@cli.command()
def version():
    """Show version information."""
    from src import __version__
    click.echo(f"RAG Platform version {__version__}")


# Import command modules
from src.cli.commands import ingest, query, index


# Register command groups
cli.add_command(ingest.ingest)
cli.add_command(query.query)
cli.add_command(index.index)


if __name__ == '__main__':
    cli()
