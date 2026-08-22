# RAG Platform

Privacy-focused Retrieval-Augmented Generation (RAG) platform that answers questions from your documents using local Ollama LLM with explicit source citations.

## Features

- **100% Local & Private** - All processing on your machine using Ollama
- **Source Citations** - Every answer includes document names and page numbers
- **Multi-Stage Retrieval** - Vector search → Re-ranking → Context assembly
- **Comprehensive PDF Analysis** - Extracts text, tables, figures, and metadata
- **Docker Support** - Easy setup for collaborators

---

## Quick Start with Docker

### Prerequisites

1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop)
2. **Ollama** - [Download here](https://ollama.ai/)

### Step 1: Install Ollama and Pull Models

```bash
# Install Ollama from https://ollama.ai/

# Pull required models (one-time, ~5 GB total)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Verify models are installed
ollama list
```

You should see:
```
NAME                       ID              SIZE
llama3.1:8b                46e0c10c039e    4.9 GB
nomic-embed-text:latest    0a109f422b47    274 MB
```

### Step 2: Build Docker Image

```bash
# Clone/navigate to project directory
cd /path/to/RAG

# Build the Docker image (first time only, ~5 minutes)
make build
```

### Step 3: Ingest Documents

Place your PDF documents in the `soc/` folder, then run:

```bash
# Index all documents (one-time, ~30 minutes for 17 PDFs)
make ingest
```

This will:
- Extract text, tables, and figures from all PDFs
- Generate embeddings for semantic search
- Build searchable index in `data/` folder

### Step 4: Query the System

```bash
# Ask questions about your documents
make query Q="What is Legionella?"
make query Q="What are water quality standards?"
make query Q="What are CDC recommendations?"

# Check index status
make status

# List all indexed documents
make list
```

---

## Docker Commands Reference

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make build` | Build Docker image |
| `make ingest` | Ingest all documents from soc/ folder |
| `make query Q="question"` | Ask a question |
| `make status` | Show index statistics |
| `make list` | List all indexed documents |
| `make shell` | Open interactive terminal in container |
| `make logs` | View application logs |
| `make clean` | Remove all data and start fresh |
| `make rebuild` | Rebuild Docker image from scratch |
| `make test-connection` | Test connection to Ollama |

---

## Example Queries

```bash
# About Legionella
make query Q="What is Legionella and how does it spread?"

# Water management
make query Q="What are water management program requirements?"

# Standards and guidelines
make query Q="What are CDC recommendations for water systems?"
make query Q="What are OSHA guidelines?"
make query Q="What are WHO water quality standards?"

# Technical details
make query Q="What testing methods are recommended?"
make query Q="What are control measures for Legionella?"
```

---

## Project Structure

```
RAG/
├── soc/                    # Your PDF documents (put PDFs here)
├── data/                   # Generated indices (auto-created, persisted)
│   ├── vector_store/       # Vector embeddings (Qdrant)
│   ├── graph_db/           # Semantic graph (NetworkX)
│   └── metadata.db         # Document metadata (SQLite)
├── logs/                   # Application logs
├── src/                    # Source code
│   ├── core/               # Configuration and logging
│   ├── document_processing/# PDF parsing and chunking
│   ├── embedding/          # Embedding generation
│   ├── indexing/           # Vector and metadata storage
│   ├── retrieval/          # Multi-stage retrieval pipeline
│   ├── generation/         # LLM response generation
│   ├── graph/              # Semantic graph (coming soon)
│   ├── cli/                # Command-line interface
│   ├── api/                # REST API (coming soon)
│   └── web/                # Web UI (coming soon)
├── config/                 # Configuration files
│   ├── models.yaml         # Model configurations
│   ├── retrieval.yaml      # Retrieval parameters
│   └── chunking.yaml       # Chunking strategies
├── Makefile                # Docker commands
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Docker image definition
└── pyproject.toml          # Python dependencies
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Machine                            │
│                                                              │
│  ┌──────────────┐              ┌──────────────────────┐    │
│  │   Ollama     │◄─────────────│  Docker Container    │    │
│  │  (LLM Host)  │              │                       │    │
│  │  :11434      │              │  RAG Platform         │    │
│  └──────────────┘              │  - PDF Processing     │    │
│                                 │  - Vector Search      │    │
│  ┌──────────────┐              │  - LLM Generation     │    │
│  │  ./soc/      │◄─────────────│                       │    │
│  │  (PDFs)      │  mounted     │  Mounted volumes:     │    │
│  └──────────────┘              │  - ./soc → /app/soc   │    │
│                                 │  - ./data → /app/data │    │
│  ┌──────────────┐              │  - ./logs → /app/logs │    │
│  │  ./data/     │◄─────────────│                       │    │
│  │  (Indices)   │  persisted   └──────────────────────┘    │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- **Ollama runs on your host machine** (better GPU access, easier setup)
- **RAG app runs in Docker container** (consistent environment)
- **Documents and data are mounted** from your machine (persisted across restarts)

---

## Answer Format

Every answer includes explicit source citations:

```
Answer: According to CDC guidelines [1], water quality testing should be
performed weekly using approved methods. OSHA standards [2] require daily
monitoring for high-risk facilities.

Sources:
[1] CDC, 2017.pdf - Pages 45-47: "Water Quality Testing Protocols"
[2] OSHA technical manual, 3, 7, 1999.pdf - Page 23: "Monitoring Requirements"
```

---

## Configuration

### Environment Variables

Create/edit `.env` file:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag.log
```

Note: retrieval sizing (`top_k`, `top_n`, `max_context_tokens`) is controlled by `config/retrieval.yaml`, not `.env` — see the Retrieval Configuration section below. `rag query --top-k`/`--top-n` override the YAML defaults for a single query.

### Model Configuration

Edit `config/models.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "llama3.1:8b"
  temperature: 0.1
  max_tokens: 2048
  context_window: 32768

embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  batch_size: 32

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  enabled: true
```

This file is loaded on every `rag ingest`/`rag query` run — editing it (e.g. to point at a different Ollama model) takes effect the next time you run either command.

### Retrieval Configuration

Edit `config/retrieval.yaml`:

```yaml
stage1_vector_search:
  top_k: 50
  distance_metric: "cosine"

stage2_reranking:
  enabled: true
  top_n: 10

stage3_graph_expansion:
  enabled: false  # Coming soon
  max_hops: 1
  max_neighbors: 10

stage4_context_assembly:
  max_context_tokens: 8192
  sort_by: ["relevance", "document", "page"]
```

Also loaded automatically. One exception: graph expansion (stage 3) isn't implemented, so `stage3_graph_expansion.enabled` is ignored (with a logged warning) regardless of what it's set to here. Use `rag query --top-k`/`--top-n` to override retrieval size for a single query without editing the file.

---

## Troubleshooting

### Can't Connect to Ollama

```bash
# Check Ollama is running
ollama list

# If not running, start it
ollama serve

# Test connection from Docker
make test-connection
```

### Need to Rebuild Index

```bash
# Clear everything and start fresh
make clean

# Rebuild from scratch
make ingest

# Or, without leaving Docker/Poetry, clear and re-ingest in one step:
rag ingest soc/ --rebuild
```

### Want to See What's Happening

```bash
# View real-time logs
make logs

# Open interactive shell inside container
make shell

# Inside container, you can run:
rag --help
rag query "your question"
rag index status
```

### Docker Issues

```bash
# Rebuild Docker image from scratch
make rebuild

# Check Docker is running
docker ps

# View Docker logs
docker-compose logs
```

### Memory Issues

If you encounter memory issues:

1. Close other applications
2. Reduce batch size in `config/models.yaml`:
   ```yaml
   embedding:
     batch_size: 16  # Default: 32
   ```
3. Process fewer documents at once

---

## Performance

### Current Scale (17 PDFs, ~56 MB)

- **Ingestion**: ~4-5 minutes measured on the author's machine (one-time; hardware-dependent)
- **Query Latency**: ~20-30 seconds per query, mostly `llama3.1:8b` generation time. Each `rag query` is its own process, so there's no persistent "warm" state — every query pays the same startup cost.
- **Memory Usage**: ~4-6 GB RAM
- **Disk Usage**: ~2 GB (indices + models)

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 5GB free space (models + indices)
- **CPU**: Any modern CPU (GPU optional, helps with Ollama)

---

## Technology Stack

- **LLM**: Ollama (llama3.1:8b) - 128k context window
- **Embeddings**: nomic-embed-text (768 dimensions)
- **Vector Store**: Qdrant (embedded mode, HNSW indexing)
- **Re-ranker**: sentence-transformers (MiniLLM-L6-v2)
- **PDF Processing**: PyMuPDF, pdfplumber, pytesseract
- **Metadata Store**: SQLite
- **Graph**: NetworkX (coming soon)
- **CLI**: Click + Rich (colored terminal output)
- **Containerization**: Docker + docker-compose

---

## Privacy & Security

- **100% Local**: All processing happens on your machine
- **No External API Calls**: Except initial model downloads
- **No Telemetry**: No usage data sent anywhere
- **Secure Storage**: All data in local filesystem
- **No Internet Required**: After initial setup (for queries)

---

## Native Setup (Alternative to Docker)

If you prefer to run without Docker:

```bash
# Install Poetry
pip install poetry

# Install dependencies
poetry install

# Activate virtual environment
source $(poetry env info --path)/bin/activate

# Ingest documents
rag ingest soc/

# Query
rag query "your question"
```

---

## Development

### Running Tests

```bash
# Enter container
make shell

# Inside container:
pytest
pytest --cov=src --cov-report=html
```

### Adding New Documents

```bash
# Add PDFs to soc/ folder
cp /path/to/new.pdf soc/

# Re-run ingestion
make ingest
```

### Viewing Logs

```bash
# Real-time logs
make logs

# Or view log file directly
tail -f logs/rag.log
```

---

## Project Status

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for implementation status and roadmap.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical architecture.

---

## License

Private project - All rights reserved.

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `make logs` or `logs/rag.log`
3. Contact project maintainer

---

**Built for privacy-focused document analysis 🔒**
