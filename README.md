# RAG Platform

Privacy-focused Retrieval-Augmented Generation (RAG) platform that answers questions from your documents using local Ollama LLM with explicit source citations.

## Features

- **Local-First & Private** - Documents, embeddings, and chat history always stay on your machine, regardless of which LLM answers a question
- **Source Citations** - Every answer includes document names, page numbers, and line ranges
- **Multi-Stage Retrieval** - Vector search → Re-ranking → Context assembly
- **Comprehensive PDF Analysis** - Extracts text, tables, figures, and metadata
- **Multi-User Web App** - Real accounts, multiple persistent chats per user, streamed answers
- **Per-Chat & Shared Documents** - Upload a PDF into one chat, or explicitly share it to your whole knowledge base
- **Bring Your Own Key (optional)** - Answer with local Ollama (default) or opt in to OpenAI/Anthropic per chat — only that chat's live prompt ever leaves the machine
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
| `make web` | Launch the web UI at http://localhost:7860 |
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
│   ├── auth/                # Multi-user accounts, sessions, API-key storage
│   ├── chat/                # Chat/message models, per-chat document scoping
│   ├── api/                 # FastAPI app: auth/chats/documents/settings routes
│   └── web/
│       └── static/          # Vanilla HTML/CSS/JS frontend (served by FastAPI)
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
[1] CDC, 2017.pdf - Page 45, Lines 12-18: "Water Quality Testing Protocols"
[2] OSHA technical manual, 3, 7, 1999.pdf - Page 23, Lines 3-9: "Monitoring Requirements"
```

Citations point to the specific lines a chunk came from, not just the page — chunks are kept small (~300 tokens) specifically so a line range stays tight rather than covering most of a page. Only sources the answer actually cites are listed; chunks retrieved but not used in the answer are left out.

---

## Web UI

Prefer a browser to the terminal? `make web` (or `rag web` outside Docker) launches a
multi-user chat platform at **http://localhost:7860**:

- **Accounts** — sign up / log in; each user has their own chats and documents.
- **Chat** — create as many persistent conversations as you like, ask questions, watch the
  answer stream in token-by-token, see exactly which sources it cited (with page and line
  numbers) underneath.
- **Documents** — upload a PDF into a chat (scoped to that chat by default) via the
  documents drawer, or check "add to my shared knowledge base" to make it visible from your
  other chats too. A chat with no visible documents says so, rather than guessing.
- **Provider** — click the provider badge in a chat's top bar to switch between local Ollama
  (default, no setup needed) and OpenAI/Anthropic, and optionally pin a specific model.
- **Settings** — add/remove your own OpenAI/Anthropic API keys (Settings button in the
  sidebar). Keys are encrypted at rest and never echoed back — the UI only shows a masked
  suffix once saved.

```bash
make web                # Docker
rag web                 # Native (Poetry)
rag web --port 8080     # Use a different port
```

**Before first run**, the app needs a `SECRET_KEY` in `.env` (used to encrypt stored provider
API keys — the app refuses to start without one):

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste the output into .env as SECRET_KEY=...
```

**Privacy model**: documents, embeddings, and chat history are always stored locally, no
matter which provider a chat uses. Choosing OpenAI or Anthropic for a chat sends only that
chat's live prompt to the provider's API for that one request — and only because you
explicitly selected that provider. Local Ollama (the default) never leaves the machine at all.

The web server keeps the embedder, re-ranker, and the default Ollama client warm for its whole
lifetime, so after the first query, subsequent ones skip that reload cost (unlike the CLI,
which is a fresh process every time). One caveat: Qdrant's embedded vector store only allows
one open client on its storage path at a time, so don't run `rag web` and `rag query`/
`rag ingest` against the same `data/` folder simultaneously — the same restriction that
already applied between `rag ingest` and `rag query`.

---

## Configuration

### Environment Variables

Create/edit `.env` file:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Web platform — required to run `rag web`, generate with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
SECRET_KEY=

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag.log
```

OpenAI/Anthropic keys are **not** set here — each user adds their own from the web app's
Settings modal, encrypted at rest in the local database rather than living in `.env`.

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

- **LLM**: Ollama (llama3.1:8b) by default — 128k context window; opt-in OpenAI or Anthropic per chat
- **Embeddings**: nomic-embed-text (768 dimensions) — always local, regardless of LLM provider choice
- **Vector Store**: Qdrant (embedded mode, HNSW indexing)
- **Re-ranker**: sentence-transformers (MiniLLM-L6-v2)
- **PDF Processing**: PyMuPDF, pdfplumber, pytesseract
- **Metadata Store**: SQLite
- **Graph**: NetworkX (coming soon)
- **CLI**: Click + Rich (colored terminal output)
- **Web platform**: FastAPI + hand-written vanilla HTML/CSS/JS (no Node toolchain), multi-user auth (bcrypt + server-side sessions), API keys encrypted at rest (Fernet)
- **Containerization**: Docker + docker-compose

---

## Privacy & Security

- **Local-First Storage**: documents, embeddings, and chat history always stay on your machine, no matter which LLM provider you pick for a chat
- **Cloud Providers Are Opt-In**: local Ollama is the default and needs no external calls at all; OpenAI/Anthropic only get that one chat's live prompt, only when you've explicitly selected them for it
- **No Telemetry**: No usage data sent anywhere
- **Secure Storage**: passwords hashed (bcrypt), provider API keys encrypted at rest (Fernet), session tokens are opaque and server-revocable (not JWT)
- **No Internet Required**: for the default local (Ollama) setup, after initial model download

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
