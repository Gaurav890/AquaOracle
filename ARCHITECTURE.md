# RAG Platform Architecture

## Overview

This document describes the architecture of the privacy-focused RAG (Retrieval-Augmented Generation) platform.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE LAYER                        │
│         CLI Commands | REST API | Web Dashboard             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│              QUERY PROCESSING & GENERATION                   │
│  Query → Retrieval Pipeline → LLM (Ollama) → Response       │
│                    with Citations                            │
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

**Current gap**: the `tables` and `figures` tables above exist in the schema, but nothing writes to them yet — `TableExtractor`/`FigureExtractor` produce real data during ingestion, only the counts make it into `documents.table_count`/`figure_count`. Treat table and figure content as not yet queryable.

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

#### REST API (TODO: `src/api/`)
- FastAPI application
- Endpoints: /api/query, /api/admin/*, /api/health
- Swagger documentation at /docs
- CORS and logging middleware

#### Web UI (TODO: `src/web/`)
- Gradio or Streamlit interface
- Chat-style query interface
- Source preview and citation display
- System dashboard

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
- **FastAPI**: REST API
- **Gradio**: Web UI

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

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag.log
```

### YAML Configurations

**models.yaml**: Model parameters
**retrieval.yaml**: Retrieval pipeline settings
**chunking.yaml**: Chunking strategies

**Current gap**: these files, plus the `VECTOR_TOP_K`/`RERANK_TOP_N`/`MAX_CONTEXT_TOKENS` env vars, aren't loaded by the running code — `ModelConfig`/`RetrievalConfig`/`ChunkingConfig` and `load_model_config()` exist in `src/core/config.py` but are never called. Every pipeline parameter is hardcoded directly in the CLI commands (`src/cli/commands/*.py`). Editing these YAML files currently has no effect; use `rag query --top-k`/`--top-n` to change retrieval size instead.

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
- 100% local processing
- No external API calls (except model downloads)
- No telemetry or tracking
- Data never leaves local machine

### Data Protection
- Local file storage with OS permissions
- No cloud storage or services
- SQLite database with file-level security

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

### Phase 2
- Complete graph-based retrieval
- REST API implementation
- Web UI development

### Phase 3
- Multi-modal support (understand tables, figures)
- Query expansion and rewriting
- Conversational mode (multi-turn)

### Phase 4
- Hybrid search (BM25 + vector)
- Advanced analytics
- Document management UI
- Export and reporting

## References

- [Ollama Documentation](https://ollama.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [sentence-transformers](https://www.sbert.net/)

---

**Last Updated**: 2026-01-29
