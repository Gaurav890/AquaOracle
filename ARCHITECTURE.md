# RAG Platform Architecture

## Overview

This document describes the architecture of the privacy-focused RAG (Retrieval-Augmented Generation) platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE LAYER                        │
│  CLI Commands | FastAPI REST/SSE API | Web App (multi-user) │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│         AUTH, CHAT & DOCUMENT SCOPING (web app only)         │
│  Session cookies → per-user chats → allowed-doc-id resolver │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│              QUERY PROCESSING & GENERATION                   │
│  Query → Retrieval Pipeline → LLM (Ollama/OpenAI/Anthropic) │
│         → Response with Citations (provider is per-chat,     │
│           chosen by the user; local Ollama by default)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│           MULTI-STAGE RETRIEVAL PIPELINE                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Vector Search (HNSW) → Top 50-100 candidates    │   │
│  │  2. Re-ranking (MiniLLM) → Top 10-20 chunks         │   │
│  │  3. Graph Expansion → Add context from linked pages │   │
│  │  4. Context Assembly → Build citation map           │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│              STORAGE & INDEX LAYER                           │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Qdrant      │  │ NetworkX    │  │ SQLite           │   │
│  │ Vector DB   │  │ Page Graph  │  │ Metadata Store   │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│         DOCUMENT PROCESSING PIPELINE                         │
│  PDF Parsing → Text/Table/Figure Extraction → Chunking      │
│  → Embedding (Ollama) → Indexing → Graph Building           │
└──────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Processing Pipeline

**Purpose**: Extract and prepare content from PDF documents

**Components**:
- **PDFParser** (`src/document_processing/pdf_parser.py`)
  - Uses PyMuPDF (fitz) for comprehensive PDF parsing
  - Extracts text, layout, page metadata
  - Preserves document structure

- **TableExtractor** (`src/document_processing/table_extractor.py`)
  - Uses pdfplumber for table detection and extraction
  - Preserves table structure (rows, columns, headers)
  - Converts tables to structured format

- **FigureExtractor** (`src/document_processing/figure_extractor.py`)
  - Extracts images and diagrams
  - Identifies figure captions
  - Stores figure metadata

- **Chunker** (`src/document_processing/chunker.py`)
  - **Strategy**: Hybrid semantic + sliding window
  - Respects natural boundaries (paragraphs, sections)
  - Max chunk size: 1024 tokens with 20% overlap
  - Preserves metadata (doc_id, page_numbers, section_title)

- **MetadataExtractor** (`src/document_processing/metadata_extractor.py`)
  - Extracts title, author, organization, year
  - Parses PDF metadata fields
  - Uses filename heuristics for missing data

**Data Flow**:
```
PDF File → PDFParser → PDFDocument
                     ↓
          [Tables, Figures, Metadata]
                     ↓
                 Chunker
                     ↓
              List of Chunks
```

### 2. Embedding & Vector Storage

**Purpose**: Generate and store embeddings for semantic search

**Components**:
- **OllamaEmbedder** (`src/embedding/ollama_embedder.py`)
  - Uses Ollama API with nomic-embed-text model
  - Batch processing for efficiency (batch_size=32)
  - 768-dimensional embeddings
  - Error handling and retry logic

- **VectorStore** (`src/indexing/vector_store.py`)
  - Qdrant vector database (embedded mode)
  - HNSW index for fast similarity search
  - Distance metric: Cosine similarity
  - Payload indexing for metadata filtering

**Storage Schema**:
```json
{
  "vector": [768 floats],
  "payload": {
    "chunk_id": "doc_id_chunk_0",
    "doc_id": "CDC_2017",
    "text": "chunk text...",
    "page_numbers": [1, 2],
    "token_count": 512
  }
}
```

### 3. Metadata Storage

**Purpose**: Store document and chunk metadata in relational format

**Database**: SQLite

**Schema**:

**documents** table:
```sql
- doc_id (PRIMARY KEY)
- file_name, file_path
- title, author, organization, year
- page_count, chunk_count, table_count, figure_count
- file_size, processed_at
- metadata_json
```

**chunks** table:
```sql
- id (AUTOINCREMENT)
- chunk_id (UNIQUE)
- doc_id (FOREIGN KEY)
- chunk_index
- text, token_count
- page_numbers (JSON array)
- section_title
- char_start, char_end
- metadata_json
- created_at
```

**Indexes**:
- `idx_chunks_doc_id` on chunks(doc_id)
- `idx_documents_org` on documents(organization)
- `idx_documents_year` on documents(year)

`TableExtractor`/`FigureExtractor` results are persisted here via `MetadataStore.add_table`/`add_figure`, queryable per-document via `get_tables`/`get_figures`. They aren't embedded into the vector store, though, so a query can't yet retrieve a table or figure by semantic similarity the way it retrieves prose chunks.

### 4. Retrieval Pipeline

**Purpose**: Multi-stage retrieval for high-quality results

**Stages**:

1. **Vector Search** (`src/retrieval/vector_retriever.py`)
   - Generate query embedding
   - Search Qdrant with HNSW
   - Retrieve top K=50-100 candidates
   - Filter by metadata if needed

2. **Re-ranking** (`src/retrieval/reranker.py`)
   - Use cross-encoder model (ms-marco-MiniLM-L-6-v2)
   - Score each (query, chunk) pair
   - Select top N=10-20 most relevant
   - More accurate than vector similarity alone

3. **Graph Expansion** (TODO)
   - From top chunks, extract source pages
   - Traverse semantic graph
   - Find related pages (k=1-2 hops)
   - Add chunks from related pages

4. **Context Assembly** (`src/retrieval/context_assembler.py`)
   - Merge chunks from all stages
   - Remove duplicates
   - Sort by: relevance → document → page
   - Trim to context window (8192 tokens)
   - Build citation map

**Retrieval Pipeline** (`src/retrieval/retrieval_pipeline.py`)
- Orchestrates all stages
- Configurable (enable/disable stages)
- Returns RetrievalResult with chunks and citation map

### 5. LLM Generation

**Purpose**: Generate answers with citations using local LLM

**Components**:

- **OllamaClient** (`src/generation/ollama_client.py`)
  - Interface to Ollama API
  - Model: llama3.1:8b
  - Temperature: 0.1 (factual responses)
  - Supports streaming and chat modes

- **Prompt Templates** (`src/generation/prompt_templates.py`)
  - **System Prompt**: Instructions for factual, cited responses
  - **RAG Prompt**: Question + context with source markers
  - Citation extraction and validation prompts

- **ResponseGenerator** (`src/generation/response_generator.py`)
  - Orchestrates retrieval + generation
  - Builds prompt with context
  - Generates answer with LLM
  - Formats sources for display
  - Returns RAGResponse with answer, sources, metadata

**Prompt Structure**:
```
Context from source documents:

[1] Source: CDC, 2017.pdf, Pages 45-47
[Text content...]

[2] Source: OSHA technical manual, Page 23
[Text content...]

---

Question: What are water quality standards?

Instructions:
- Answer using ONLY the context
- Cite sources using [1], [2], etc.
- If not in context, say so

Answer:
```

### 6. User Interfaces

#### CLI (`src/cli/`)
- **Commands**: ingest, query, index
- Built with Click
- Rich terminal output with colors and progress bars
- Entry point: `rag` command

**Example**:
```bash
rag ingest soc/
rag query "What are CDC water quality standards?"
rag index status
```

#### Web platform (`src/api/` + `src/web/static/`)

The web UI was originally a single-user Gradio app. It has been rebuilt into a multi-user
platform: FastAPI backend (`src/api/`) serving a hand-written vanilla HTML/CSS/JS frontend
(`src/web/static/` — no Node toolchain, no React/Vue/Svelte, styled in a dark/sharp-edged
Linear/Vercel look). The old Gradio app (`src/web/app.py`, `theme.py`) and the `gradio`
dependency have both been removed.

- **App factory** (`src/api/main.py`): `create_app()` registers the auth/chats/documents/settings
  routers, mounts the static frontend, and refuses to start if `SECRET_KEY` isn't set (needed to
  encrypt stored provider API keys). Launched via `rag web` (`src/cli/commands/web.py`, now a thin
  `uvicorn.run(...)` wrapper) or `make web`, served at `:7860`.
- **Auth** (`src/auth/`, `src/api/routes/auth.py`): real signup/login/logout. Passwords hashed
  with `bcrypt`. Sessions are opaque `secrets.token_urlsafe(32)` tokens stored server-side in a
  `sessions` table and set as an `httponly`/`samesite=Lax` cookie — deliberately not JWT (JWT
  revocation needs a blocklist anyway for a single local server) and not Starlette's built-in
  `SessionMiddleware` (that round-trips the whole session payload in a signed cookie and can't be
  revoked server-side without rotating the app secret for every user). `httponly` also means the
  cookie is unreadable from page JavaScript — verified during Phase 3 testing, which is why live
  endpoint verification for this rebuild is done via a real login (curl/httpx capturing its own
  `Set-Cookie`) rather than replaying a token pulled out of `document.cookie`.
- **Chats** (`src/chat/`, `src/api/routes/chats.py`): each user has multiple persistent `Chat`
  rows, each with its own `Message` history and its own `provider`/`model` (default: Ollama).
  Sending a message streams the answer over SSE (`src/api/services/streaming.py`, using the same
  `on_token` callback contract `ResponseGenerator.generate()` already exposed) and persists both
  the user's message and the assistant's reply — including on failure, so a reload always shows
  what happened.
- **Document scoping** (`src/chat/scoping.py`, `src/api/routes/documents.py`): a document uploaded
  inside a chat is scoped to that chat by default (`ChatDocument` junction table); an explicit
  "add to shared knowledge base" toggle (`documents.is_shared`) makes it visible from all of that
  user's other chats too. `get_allowed_doc_ids()` resolves one chat's visible set as: its own
  chat-scoped uploads, ∪ that user's shared docs, ∪ legacy/CLI-ingested docs (`owner_user_id IS
  NULL`, which stay visible to everyone — this preserves `rag ingest`'s existing behavior with
  zero CLI changes). That id list is passed to `VectorStore.search()` as a `doc_filter`, which
  Qdrant applies as `MatchAny` (OR-match across the list) rather than the single-value
  `MatchValue` equality filter it used before this rebuild. A chat with nothing visible
  short-circuits with a friendly note instead of querying Qdrant with an empty filter.
- **Multi-provider generation, BYOK** (`src/generation/base_client.py`,
  `provider_factory.py`, `src/api/routes/settings.py`): `OllamaClient`/`OpenAIClient`/
  `AnthropicClient` all implement one small `LLMClient` interface (`generate`/`generate_stream`/
  `.model` — the entire surface `ResponseGenerator` actually calls). `create_llm_client()`
  dispatches on the chat's `provider`, raising a clear `MissingApiKeyError` (surfaced as a normal
  in-chat error, not a crash) rather than letting a cloud SDK throw its own exception when no key
  is on file. A user's OpenAI/Anthropic key is entered once in the Settings modal, encrypted at
  rest with Fernet (`src/core/crypto.py`, keyed by `SECRET_KEY`), and decrypted server-side only
  for the duration of one generation call. **Only the live prompt for a chat's selected cloud
  provider ever leaves the machine, and only when that provider is explicitly selected** —
  documents, embeddings, and chat history are never sent anywhere regardless of provider choice.
- Shares ingestion logic with the CLI through `src/document_processing/ingestion_service.py`
  (now parameterized with `owner_user_id`/`is_shared`, defaulting to today's CLI behavior), so
  upload-triggered ingestion and `rag ingest` can't drift apart.
- Keeps the embedder, re-ranker, and the default Ollama LLM client warm for the server's
  lifetime, but opens and closes a fresh `VectorStore` per request (see Concurrency note below)
  — Qdrant's embedded mode only allows one open client on its storage path at a time. Cloud
  provider clients (`OpenAIClient`/`AnthropicClient`) are built fresh per request instead, since
  they may need a different decrypted API key per call and don't do `OllamaClient`'s eager
  connection test (which would otherwise cost a billed API call just to construct the object).

**Concurrency note**: because of that single-client constraint, the web server and the `rag`
CLI (`ingest`/`query`) shouldn't be run against the same `data/vector_store` at the same time.

## Technology Stack

### Core
- **Python 3.9+**
- **Poetry**: Dependency management

### LLM & Embeddings
- **Ollama**: Local LLM infrastructure
- **llama3.1:8b**: Text generation (4.7 GB)
- **nomic-embed-text**: Embeddings (768 dim, 274 MB)

### Storage
- **Qdrant**: Vector database (embedded mode)
- **SQLite**: Metadata storage
- **NetworkX**: Semantic graph (TODO)

### Document Processing
- **PyMuPDF (fitz)**: PDF parsing
- **pdfplumber**: Table extraction
- **pytesseract**: OCR (optional)
- **tiktoken**: Token counting

### Retrieval & ML
- **sentence-transformers**: Re-ranking models
- **transformers**: ML model infrastructure
- **torch**: Deep learning backend

### UI & CLI
- **Click**: CLI framework
- **Rich**: Terminal formatting
- **FastAPI**: REST/SSE API + static frontend host
### Auth & Security
- **bcrypt**: Password hashing
- **cryptography (Fernet)**: Encryption at rest for stored provider API keys
- **email-validator**: Pydantic `EmailStr` validation for signup

### Cloud LLM Providers (opt-in BYOK)
- **openai**: OpenAI chat completions (`OpenAIClient`)
- **anthropic**: Anthropic messages API (`AnthropicClient`) — pinned `^0.40.0`; 0.39.x is
  incompatible with the httpx version already pinned transitively by other deps (it passes a
  `proxies` kwarg httpx 0.28+ rejects)

### Utilities
- **loguru**: Structured logging
- **pydantic**: Configuration management
- **tqdm**: Progress bars

## Configuration

### Environment Variables (`.env`)
```env
# Paths
PROJECT_ROOT=/Users/gaurav/PROJECTS/RAG
SOURCE_DOCS_PATH=soc
DATA_PATH=data

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# Retrieval
VECTOR_TOP_K=50
RERANK_TOP_N=10
MAX_CONTEXT_TOKENS=8192

# API
API_HOST=0.0.0.0
API_PORT=8000

# Auth & Security (web platform) — required, the app refuses to start without it
# Generate with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
SECRET_KEY=

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag.log
```

### YAML Configurations

**models.yaml**: Model parameters
**retrieval.yaml**: Retrieval pipeline settings
**chunking.yaml**: Chunking strategies

These are loaded via `load_model_config()`/`load_retrieval_config()`/`load_chunking_config()` in `src/core/config.py` and drive the CLI commands (`src/cli/commands/*.py`) — editing them changes ingestion/retrieval behavior on the next run. The one exception is `stage3_graph_expansion.enabled`: since graph expansion isn't implemented, the app logs a warning and ignores that flag rather than acting on it. `rag query --top-k`/`--top-n` still override the YAML defaults per-invocation.

## Data Flow

### Ingestion
```
PDF File
  ↓
PDFParser → [Document, Pages]
  ↓
Extractors → [Tables, Figures]
  ↓
Chunker → [Chunks with metadata]
  ↓
Embedder → [Embeddings]
  ↓
Storage → [Qdrant + SQLite]
```

### Query
```
User Question
  ↓
Embedder → Query Embedding
  ↓
Vector Search → Top K candidates
  ↓
Re-ranker → Top N chunks
  ↓
Graph Expansion → Add context
  ↓
Context Assembly → Final context + citation map
  ↓
LLM Generator → Answer with citations
  ↓
User Response
```

## Performance Characteristics

### Current Scale (17 PDFs, ~56 MB)
- **Ingestion**: ~4-5 minutes measured on the author's machine (one-time; hardware-dependent)
- **Query Latency**: ~20-30 seconds end-to-end, dominated by `llama3.1:8b` generation time. Since each `rag query` is its own CLI process, it also re-tests the Ollama connection and reloads the re-ranker model every time — there's no persistent "warm" state between queries.
- **Memory**: ~4 GB RAM
- **Disk**: ~500 MB (indices + metadata)

### Optimizations
1. **Batch Embedding**: Process 32 chunks at once
2. **HNSW Index**: O(log N) search complexity
3. **Caching**: Query results, embeddings
4. **Lazy Loading**: Load models on demand

## Security & Privacy

### Privacy Guarantees
- All storage — documents, embeddings, chat history, and stored provider API keys — stays local
  regardless of which LLM provider a chat uses
- Local Ollama is the default provider and requires no external API calls at all
- OpenAI/Anthropic are strictly opt-in per chat (BYOK): choosing one sends **only that request's
  live prompt** to the provider's API — never documents, embeddings, or other chats' history —
  and only for the duration of that one generation call
- No telemetry or tracking

### Data Protection
- Local file storage with OS permissions
- No cloud storage or services (aside from a user's own explicitly-chosen cloud LLM provider)
- SQLite database with file-level security
- Passwords hashed with bcrypt, never stored in plaintext
- Session tokens are opaque and server-revocable (not JWT) — logout deletes the session row, so
  no app-wide secret rotation is needed to invalidate one user's session
- Provider API keys encrypted at rest (Fernet, keyed by `SECRET_KEY`) and never echoed back to
  the client once stored — the Settings UI shows only a masked suffix

## Scalability

### Current Limits
- ~100 documents: Single machine with 8GB RAM
- ~1000 documents: 16GB RAM, SSD recommended
- ~10000 documents: 32GB RAM, dedicated vector DB server

### Scaling Options
1. **Qdrant Server Mode**: Separate vector DB server
2. **PostgreSQL**: Replace SQLite for metadata
3. **Neo4j**: Replace NetworkX for graph
4. **Distributed Processing**: Multi-node ingestion
5. **GPU Acceleration**: Faster embedding generation

## Testing Strategy

### Unit Tests
- Component-level testing
- Mock external dependencies
- Fast execution (<10s)

### Integration Tests
- End-to-end pipeline testing
- Real dependencies
- Sample document corpus

### Performance Tests
- Query latency benchmarks
- Retrieval quality metrics (precision@k, MRR)
- Memory usage profiling

## Future Enhancements

### Later
- Graph-based retrieval
- Multi-modal support (retrieve tables/figures by semantic similarity, not just prose)
- Query expansion and rewriting
- Hybrid search (BM25 + vector)
- Advanced analytics, export and reporting

## References

- [Ollama Documentation](https://ollama.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [sentence-transformers](https://www.sbert.net/)

---

**Last Updated**: 2026-09-07
