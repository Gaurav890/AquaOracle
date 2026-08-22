"""SQLite metadata store for documents and chunks."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sqlite3
import json
from loguru import logger


class MetadataStore:
    """SQLite-based metadata storage for documents and chunks."""

    def __init__(self, db_path: Path):
        """
        Initialize metadata store.

        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logger.bind(name="MetadataStore")
        self.db_path = db_path

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    title TEXT,
                    author TEXT,
                    organization TEXT,
                    year INTEGER,
                    page_count INTEGER,
                    chunk_count INTEGER DEFAULT 0,
                    table_count INTEGER DEFAULT 0,
                    figure_count INTEGER DEFAULT 0,
                    file_size INTEGER,
                    processed_at TIMESTAMP,
                    metadata_json TEXT,
                    UNIQUE(file_path)
                )
            """)

            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER,
                    page_numbers TEXT,
                    section_title TEXT,
                    char_start INTEGER,
                    char_end INTEGER,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
            """)

            # Tables table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    row_count INTEGER,
                    col_count INTEGER,
                    has_header BOOLEAN,
                    table_data_json TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
            """)

            # Figures table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS figures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    image_index INTEGER,
                    caption TEXT,
                    width REAL,
                    height REAL,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_org ON documents(organization)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year)")

            conn.commit()

        self.logger.info(f"Initialized metadata store at {self.db_path}")

    def add_document(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Add or update document metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO documents (
                        doc_id, file_name, file_path, title, author,
                        organization, year, page_count, file_size,
                        processed_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    metadata.get("file_name"),
                    metadata.get("file_path"),
                    metadata.get("title"),
                    metadata.get("author"),
                    metadata.get("organization"),
                    metadata.get("year"),
                    metadata.get("page_count", 0),
                    metadata.get("file_size", 0),
                    metadata.get("processed_at", datetime.now().isoformat()),
                    json.dumps(metadata),
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to add document {doc_id}: {e}")
            return False

    def add_chunk(self, chunk_id: str, doc_id: str, chunk_data: Dict[str, Any]) -> bool:
        """Add chunk metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                page_numbers_str = json.dumps(chunk_data.get("page_numbers", []))

                cursor.execute("""
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, chunk_index, text, token_count,
                        page_numbers, section_title, char_start, char_end,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id,
                    doc_id,
                    chunk_data.get("chunk_index", 0),
                    chunk_data.get("text", ""),
                    chunk_data.get("token_count", 0),
                    page_numbers_str,
                    chunk_data.get("section_title"),
                    chunk_data.get("char_start", 0),
                    chunk_data.get("char_end", 0),
                    json.dumps(chunk_data.get("metadata", {})),
                ))

                # Update chunk count in documents table
                cursor.execute("""
                    UPDATE documents
                    SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE doc_id = ?)
                    WHERE doc_id = ?
                """, (doc_id, doc_id))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to add chunk {chunk_id}: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
                row = cursor.fetchone()

                if row:
                    return dict(row)

        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")

        return None

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
                row = cursor.fetchone()

                if row:
                    data = dict(row)
                    data["page_numbers"] = json.loads(data.get("page_numbers", "[]"))
                    data["metadata"] = json.loads(data.get("metadata_json", "{}"))
                    return data

        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {e}")

        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM documents ORDER BY processed_at DESC")
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                # Document count
                cursor.execute("SELECT COUNT(*) FROM documents")
                stats["document_count"] = cursor.fetchone()[0]

                # Chunk count
                cursor.execute("SELECT COUNT(*) FROM chunks")
                stats["chunk_count"] = cursor.fetchone()[0]

                # Table count
                cursor.execute("SELECT COUNT(*) FROM tables")
                stats["table_count"] = cursor.fetchone()[0]

                # Figure count
                cursor.execute("SELECT COUNT(*) FROM figures")
                stats["figure_count"] = cursor.fetchone()[0]

                # Total pages
                cursor.execute("SELECT SUM(page_count) FROM documents")
                stats["total_pages"] = cursor.fetchone()[0] or 0

                return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

    def clear_all(self) -> bool:
        """Delete all documents, chunks, tables, and figures."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks")
                cursor.execute("DELETE FROM tables")
                cursor.execute("DELETE FROM figures")
                cursor.execute("DELETE FROM documents")
                conn.commit()
                self.logger.info("Cleared all documents, chunks, tables, and figures")
                return True

        except Exception as e:
            self.logger.error(f"Failed to clear metadata store: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all associated data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM tables WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM figures WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

                conn.commit()
                self.logger.info(f"Deleted document {doc_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
