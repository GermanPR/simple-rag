"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    id: int
    filename: str
    pages: int
    uploaded_at: Optional[datetime] = None


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""
    documents: List[DocumentInfo]
    chunks_indexed: int
    success: bool = True
    message: Optional[str] = None


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of results to return")
    rerank_k: int = Field(default=20, ge=1, le=100, description="Number of candidates to consider for reranking")
    threshold: float = Field(default=0.18, ge=0.0, le=1.0, description="Minimum semantic similarity threshold")
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Semantic vs keyword weight (0-1)")
    lambda_param: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="MMR relevance vs diversity weight (0-1)")
    use_mmr: bool = Field(default=True, description="Whether to apply MMR diversification")


class CitationInfo(BaseModel):
    """Citation information for a retrieved chunk."""
    chunk_id: int
    filename: str
    page: int
    snippet: str = Field(..., max_length=200, description="Short text snippet from the chunk")


class DebugInfo(BaseModel):
    """Debug information about the retrieval process."""
    semantic_top1: float = Field(..., description="Top-1 semantic similarity score")
    alpha: float = Field(..., description="Alpha parameter used for fusion")
    lambda_param: float = Field(..., description="Lambda parameter used for MMR")
    total_chunks: int = Field(..., description="Total number of chunks considered")
    keyword_results: int = Field(..., description="Number of keyword search results")
    semantic_results: int = Field(..., description="Number of semantic search results")
    fused_results: int = Field(..., description="Number of results after fusion")


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""
    answer: str
    citations: List[CitationInfo]
    insufficient_evidence: bool = Field(default=False)
    intent: str = Field(default="qa", description="Detected query intent")
    debug: Optional[DebugInfo] = None
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    database_connected: bool
    mistral_api_configured: bool
    total_documents: int
    total_chunks: int