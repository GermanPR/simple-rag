"""FastAPI application factory and main entry point."""

import logging
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import collections
from app.api import ingest
from app.api import query
from app.core.config import config
from app.core.logging_config import setup_logging
from app.core.models import HealthResponse
from app.core.monitoring import metrics_collector
from app.retriever.index import AsyncDatabaseManager

# Setup centralized logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting Simple RAG API server")
    logger.info(f"Database path: {config.DB_PATH}")
    logger.info(f"Mistral API configured: {config.validate_mistral_config()}")

    yield

    # Shutdown
    logger.info("Shutting down Simple RAG API server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Simple RAG API",
        description="A Retrieval-Augmented Generation system for PDF documents using FastAPI, SQLite, and Mistral AI",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on your needs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(collections.router, tags=["collections"])
    app.include_router(ingest.router, tags=["ingestion"])
    app.include_router(query.router, tags=["query"])

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Simple RAG API",
            "version": "1.0.0",
            "description": "A Retrieval-Augmented Generation system for PDF documents",
            "endpoints": {
                "health": "/health",
                "collections": "/collections",
                "ingest": "/ingest",
                "query": "/query",
                "docs": "/docs",
            },
        }

    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics."""
        return {
            "performance_stats": metrics_collector.get_all_stats(),
            "metrics_enabled": metrics_collector.enabled,
        }

    @app.post("/metrics/clear")
    async def clear_metrics():
        """Clear all collected metrics."""
        metrics_collector.clear_metrics()
        return {"message": "Metrics cleared"}

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        try:
            db_manager = AsyncDatabaseManager(config.DB_PATH)
            await db_manager.init_database()

            # Check database connectivity
            async with aiosqlite.connect(db_manager.db_path) as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM documents")
                total_docs_result = await cursor.fetchone()
                total_docs = total_docs_result[0] if total_docs_result else 0

                cursor = await conn.execute("SELECT COUNT(*) FROM chunks")
                total_chunks_result = await cursor.fetchone()
                total_chunks = total_chunks_result[0] if total_chunks_result else 0

            return HealthResponse(
                status="healthy" if config.validate_mistral_config() else "degraded",
                database_connected=True,
                mistral_api_configured=config.validate_mistral_config(),
                total_documents=total_docs,
                total_chunks=total_chunks,
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                database_connected=False,
                mistral_api_configured=config.validate_mistral_config(),
                total_documents=0,
                total_chunks=0,
            )

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        """Handle 404 errors with JSON response."""
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Not Found",
                "message": f"The requested endpoint {request.url.path} was not found",
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request, exc):
        """Handle 500 errors with JSON response."""
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal Server Error",
                "message": "An internal server error occurred",
            },
        )

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
