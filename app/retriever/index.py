"""SQLite database operations for the RAG system."""

import sqlite3
import struct
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np


class DatabaseManager:
    """Manages SQLite database operations for the RAG system."""
    
    def __init__(self, db_path: str = "rag.sqlite3"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with the required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    page INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    vec BLOB NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS df (
                    token TEXT PRIMARY KEY,
                    df INTEGER NOT NULL DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    tf REAL NOT NULL,
                    PRIMARY KEY (token, chunk_id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page);
                CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token);
                CREATE INDEX IF NOT EXISTS idx_tokens_chunk_id ON tokens(chunk_id);
            """)
    
    def add_document(self, filename: str, pages: int) -> int:
        """Add a new document and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO documents (filename, pages) VALUES (?, ?)",
                (filename, pages)
            )
            return cursor.lastrowid
    
    def add_chunk(self, doc_id: int, page: int, position: int, text: str) -> int:
        """Add a new chunk and return its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO chunks (doc_id, page, position, text) VALUES (?, ?, ?, ?)",
                (doc_id, page, position, text)
            )
            return cursor.lastrowid
    
    def store_embedding(self, chunk_id: int, embedding: np.ndarray):
        """Store L2-normalized embedding as BLOB."""
        # Normalize to L2 norm
        normalized = embedding / np.linalg.norm(embedding)
        # Convert to float32 and serialize
        blob = normalized.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vec) VALUES (?, ?)",
                (chunk_id, blob)
            )
    
    def get_embedding(self, chunk_id: int) -> Optional[np.ndarray]:
        """Retrieve embedding for a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT vec FROM embeddings WHERE chunk_id = ?", (chunk_id,)
            )
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
            return None
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """Get all embeddings as (chunk_id, embedding) pairs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT chunk_id, vec FROM embeddings")
            return [
                (chunk_id, np.frombuffer(vec, dtype=np.float32))
                for chunk_id, vec in cursor.fetchall()
            ]
    
    def store_tokens(self, chunk_id: int, token_freqs: Dict[str, float]):
        """Store token frequencies for a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert tokens for this chunk
            for token, tf in token_freqs.items():
                conn.execute(
                    "INSERT OR REPLACE INTO tokens (token, chunk_id, tf) VALUES (?, ?, ?)",
                    (token, chunk_id, tf)
                )
            
            # Update document frequencies
            for token in token_freqs.keys():
                conn.execute("""
                    INSERT OR REPLACE INTO df (token, df) 
                    VALUES (?, COALESCE((
                        SELECT COUNT(DISTINCT chunk_id) 
                        FROM tokens 
                        WHERE token = ?
                    ), 0))
                """, (token, token))
    
    def get_token_stats(self, token: str) -> Tuple[int, int]:
        """Get document frequency and total chunks for IDF calculation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT df FROM df WHERE token = ?", (token,)
            )
            row = cursor.fetchone()
            df = row[0] if row else 0
            
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            return df, total_chunks
    
    def get_chunk_tokens(self, chunk_id: int) -> Dict[str, float]:
        """Get all tokens and their TF scores for a chunk."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT token, tf FROM tokens WHERE chunk_id = ?", (chunk_id,)
            )
            return dict(cursor.fetchall())
    
    def search_chunks_by_tokens(self, tokens: List[str], limit: int = 20) -> List[Tuple[int, float]]:
        """Search chunks by tokens using TF-IDF scoring."""
        if not tokens:
            return []
        
        placeholders = ",".join("?" for _ in tokens)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT 
                    t.chunk_id,
                    SUM(t.tf * LOG((SELECT COUNT(*) FROM chunks) / (1.0 + d.df))) as score
                FROM tokens t
                JOIN df d ON t.token = d.token
                WHERE t.token IN ({placeholders})
                GROUP BY t.chunk_id
                ORDER BY score DESC
                LIMIT ?
            """, tokens + [limit])
            return cursor.fetchall()
    
    def get_chunk_info(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk information including document details."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT c.id, c.text, c.page, c.position, d.filename
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "chunk_id": row[0],
                    "text": row[1],
                    "page": row[2],
                    "position": row[3],
                    "filename": row[4]
                }
            return None
    
    def get_chunks_info(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Get information for multiple chunks."""
        if not chunk_ids:
            return []
        
        placeholders = ",".join("?" for _ in chunk_ids)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT c.id, c.text, c.page, c.position, d.filename
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.id IN ({placeholders})
            """, chunk_ids)
            return [
                {
                    "chunk_id": row[0],
                    "text": row[1],
                    "page": row[2],
                    "position": row[3],
                    "filename": row[4]
                }
                for row in cursor.fetchall()
            ]