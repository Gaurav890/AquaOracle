"""Tests for settings and the config/*.yaml loaders."""

from src.core.config import (
    Settings,
    load_chunking_config,
    load_model_config,
    load_retrieval_config,
)


def test_project_root_self_locates_to_repo_root():
    """Regression test: project_root used to default to a hardcoded absolute
    path specific to one machine, breaking for any other collaborator."""
    root = Settings().project_root

    assert (root / "pyproject.toml").exists()
    assert (root / "config").is_dir()


def test_load_model_config_reads_real_yaml():
    config = load_model_config()

    assert config.embed_model == "nomic-embed-text"
    assert config.embed_dimensions == 768
    assert config.llm_model == "llama3.1:8b"


def test_load_retrieval_config_reads_real_yaml():
    config = load_retrieval_config()

    assert config.vector_top_k == 100
    assert config.rerank_top_n == 25
    assert config.max_context_tokens == 8192


def test_load_chunking_config_reads_real_yaml():
    config = load_chunking_config()

    assert config.max_chunk_size == 300
    assert config.min_chunk_size == 40
    assert config.overlap == 0.2


def test_loaders_fall_back_to_defaults_when_yaml_missing(tmp_path):
    empty_dir = tmp_path / "no-config-here"

    model_config = load_model_config(empty_dir)
    retrieval_config = load_retrieval_config(empty_dir)
    chunking_config = load_chunking_config(empty_dir)

    assert model_config.embed_model == "nomic-embed-text"
    assert retrieval_config.vector_top_k == 100
    assert chunking_config.max_chunk_size == 300


def test_load_retrieval_config_respects_custom_yaml(tmp_path):
    (tmp_path / "retrieval.yaml").write_text(
        "stage1_vector_search:\n  top_k: 99\n"
        "stage2_reranking:\n  top_n: 3\n"
    )

    config = load_retrieval_config(tmp_path)

    assert config.vector_top_k == 99
    assert config.rerank_top_n == 3
