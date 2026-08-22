"""Query CLI commands."""

import click
from rich.console import Console
from rich.panel import Panel
from loguru import logger

from src.core.config import settings, load_model_config, load_retrieval_config
from src.embedding.ollama_embedder import OllamaEmbedder
from src.indexing.vector_store import VectorStore
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_assembler import ContextAssembler
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.generation.ollama_client import OllamaClient
from src.generation.response_generator import ResponseGenerator


console = Console()
model_config = load_model_config()
retrieval_config = load_retrieval_config()

if retrieval_config.graph_expansion_enabled:
    logger.warning(
        "retrieval.yaml sets stage3_graph_expansion.enabled=true, but graph "
        "expansion isn't implemented yet — ignoring."
    )


@click.command()
@click.argument('question', required=False)
@click.option('--top-k', default=retrieval_config.vector_top_k, help='Number of candidates from vector search')
@click.option('--top-n', default=retrieval_config.rerank_top_n, help='Number of final chunks after re-ranking')
@click.option('--no-rerank', is_flag=True, help='Disable re-ranking')
def query(question, top_k, top_n, no_rerank):
    """
    Query the RAG system.

    Examples:
        rag query "What are CDC water quality standards?"
        rag query "What are water quality standards?" --top-k 100 --top-n 20
    """
    if not question:
        console.print("[yellow]Interactive query mode not yet implemented.[/yellow]")
        console.print("[yellow]Please provide a question as an argument.[/yellow]")
        return

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")

    try:
        with console.status("[bold green]Processing query...", spinner="dots"):
            # Initialize components
            embedder = OllamaEmbedder(
                model=settings.ollama_embed_model,
                host=settings.ollama_host,
                batch_size=model_config.embed_batch_size,
            )

            vector_store = VectorStore(
                path=settings.full_vector_store_path,
                embedding_dim=model_config.embed_dimensions,
            )

            vector_retriever = VectorRetriever(
                vector_store=vector_store,
                embedder=embedder,
                top_k=top_k,
            )

            # Graph expansion (retrieval.yaml stage3) isn't implemented; only
            # combine the CLI flag with the config's rerank toggle.
            rerank_enabled = (not no_rerank) and retrieval_config.rerank_enabled
            reranker = Reranker(model_name=model_config.reranker_model, top_n=top_n) if rerank_enabled else None

            context_assembler = ContextAssembler(
                max_context_tokens=retrieval_config.max_context_tokens,
            )

            retrieval_pipeline = RetrievalPipeline(
                vector_retriever=vector_retriever,
                reranker=reranker,
                context_assembler=context_assembler,
                rerank_enabled=rerank_enabled,
                graph_expansion_enabled=False,
            )

            llm_client = OllamaClient(
                model=settings.ollama_llm_model,
                host=settings.ollama_host,
                temperature=model_config.llm_temperature,
                max_tokens=model_config.llm_max_tokens,
            )

            response_gen = ResponseGenerator(
                retrieval_pipeline=retrieval_pipeline,
                llm_client=llm_client,
            )

            # Generate response
            response = response_gen.generate(
                question=question,
                top_k=top_k,
                top_n=top_n,
            )

        # Display answer
        console.print(Panel(
            response.answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))

        # Display sources
        console.print("\n[bold cyan]Sources:[/bold cyan]")
        console.print(response.formatted_sources)

        # Display metadata
        console.print(f"\n[dim]Retrieved {response.metadata['retrieved_chunks']} chunks, "
                     f"{response.metadata['total_tokens']} tokens[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Query failed")
        raise click.Abort()
