"""CLI commands for database inspection and debug queries."""

import sqlite3
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.core.config import config
from app.llm.mistral_client import EmbeddingManager
from app.llm.mistral_client import MistralClient
from app.llm.prompts import prompt_manager
from app.logic.intent import IntentDetector
from app.logic.postprocess import answer_postprocessor

# Define simple data classes for CLI use
from app.retriever.fusion import HybridRetriever
from app.retriever.index import DatabaseManager
from app.retriever.keyword import KeywordSearcher
from app.retriever.semantic import SemanticSearcher

# Constants
CONTENT_PREVIEW_LENGTH = 200
CHUNK_PREVIEW_LENGTH = 150

console = Console()
app = typer.Typer(help="RAG System CLI Commands")


@app.command()
def chunks(
    document: str | None = typer.Option(
        None, "--doc", "-d", help="Filter by document filename"
    ),
    page: int | None = typer.Option(None, "--page", "-p", help="Filter by page number"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of chunks to show"
    ),
    content: bool = typer.Option(False, "--content", "-c", help="Show chunk content"),
    db_path: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Inspect chunks in the database."""
    db_path = db_path or config.DB_PATH

    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
    SELECT c.id as chunk_id, c.doc_id as document_id, d.filename, c.page,
           c.position, LENGTH(c.text) as content_length, c.text as content
    FROM chunks c
    JOIN documents d ON c.doc_id = d.id
    """

    params = []
    conditions = []

    if document:
        conditions.append("d.filename LIKE ?")
        params.append(f"%{document}%")

    if page is not None:
        conditions.append("c.page = ?")
        params.append(page)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY d.filename, c.page, c.position LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        console.print("[yellow]No chunks found matching criteria[/yellow]")
        return

    table = Table(title=f"Database Chunks ({len(rows)} found)")
    table.add_column("ID", style="cyan")
    table.add_column("Document", style="green")
    table.add_column("Page", style="yellow")
    table.add_column("Position", style="blue")
    table.add_column("Length", style="magenta")

    if content:
        table.add_column("Content Preview", style="white")

    for row in rows:
        content_preview = ""
        if content:
            content_text = (
                row["content"][:CONTENT_PREVIEW_LENGTH] + "..."
                if len(row["content"]) > CONTENT_PREVIEW_LENGTH
                else row["content"]
            )
            content_preview = content_text.replace("\n", " ")

        table.add_row(
            str(row["chunk_id"]),
            row["filename"],
            str(row["page"]) if row["page"] else "N/A",
            str(row["position"]),
            str(row["content_length"]),
            content_preview if content else None,
        )

    console.print(table)

    # Show summary
    cursor = conn.execute("SELECT COUNT(*) as total_chunks FROM chunks")
    total = cursor.fetchone()["total_chunks"]
    cursor = conn.execute("SELECT COUNT(DISTINCT doc_id) as total_docs FROM chunks")
    total_docs = cursor.fetchone()["total_docs"]

    console.print(
        f"\n[dim]Database summary: {total} total chunks across {total_docs} documents[/dim]"
    )
    conn.close()


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    k: int = typer.Option(5, "--k", help="Number of chunks to retrieve"),
    alpha: float = typer.Option(0.7, "--alpha", help="Semantic weight in fusion (0-1)"),
    mmr_lambda: float = typer.Option(
        0.7, "--mmr-lambda", help="MMR diversity parameter (0-1)"
    ),
    threshold: float = typer.Option(0.18, "--threshold", help="Evidence threshold"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Skip LLM generation, show only retrieval"
    ),
    db_path: str | None = typer.Option(None, "--db", help="Database path"),
):
    """Run a debug query showing detailed retrieval information."""
    db_path = db_path or config.DB_PATH

    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        sys.exit(1)

    if not config.MISTRAL_API_KEY:
        console.print("[red]MISTRAL_API_KEY not set[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold]Query:[/bold] {question}", title="Debug Query"))

    try:
        # Initialize components
        mistral_client = MistralClient()
        db_manager = DatabaseManager(db_path)
        embedding_manager = EmbeddingManager(mistral_client)
        keyword_searcher = KeywordSearcher(db_manager)
        semantic_searcher = SemanticSearcher(db_manager)

        # Create retriever with all required components
        retriever = HybridRetriever(
            db_manager=db_manager,
            keyword_searcher=keyword_searcher,
            semantic_searcher=semantic_searcher,
            alpha=alpha,
            lambda_param=mmr_lambda,
        )

        # Get query embedding
        query_embedding = embedding_manager.embed_query(question)

        # Retrieve chunks
        chunk_results = retriever.retrieve(
            query=question,
            query_embedding=query_embedding,
            top_k=k,
            rerank_k=k * 3,  # Get more candidates for reranking
            threshold=threshold,
            use_mmr=True,
        )

        # Convert to simple format for display
        chunks = []
        for chunk_id, score, metadata in chunk_results:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "content": metadata.get("text", ""),
                    "document_filename": metadata.get("filename", ""),
                    "page_number": metadata.get("page"),
                }
            )

        if not chunks:
            console.print("[yellow]No chunks retrieved[/yellow]")
            return

        # Show retrieval results
        table = Table(title=f"Retrieved Chunks (k={k}, α={alpha}, λ={mmr_lambda})")
        table.add_column("Rank", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Document", style="yellow")
        table.add_column("Page", style="blue")
        table.add_column("Content Preview", style="white")

        for i, chunk in enumerate(chunks, 1):
            content_preview = (
                chunk["content"][:CHUNK_PREVIEW_LENGTH] + "..."
                if len(chunk["content"]) > CHUNK_PREVIEW_LENGTH
                else chunk["content"]
            )
            content_preview = content_preview.replace("\n", " ")

            table.add_row(
                str(i),
                f"{chunk['score']:.4f}",
                chunk["document_filename"],
                str(chunk["page_number"]) if chunk["page_number"] else "N/A",
                content_preview,
            )

        console.print(table)

        # Check evidence threshold
        best_score = chunks[0]["score"] if chunks else 0
        has_evidence = best_score >= threshold

        console.print(
            f"\n[dim]Best score: {best_score:.4f}, Threshold: {threshold:.4f}[/dim]"
        )

        if not has_evidence:
            console.print(
                f"[red]Evidence threshold not met (score < {threshold})[/red]"
            )
            if not no_llm:
                return
        else:
            console.print(
                f"[green]Evidence threshold met (score >= {threshold})[/green]"
            )

        if no_llm:
            return

        # Generate answer
        console.print("\n" + "=" * 50)
        console.print("[bold]Generating answer...[/bold]")

        intent_detector = IntentDetector()

        # Detect intent
        intent = intent_detector.detect_intent(question)
        console.print(f"[dim]Detected intent: {intent}[/dim]")

        if not has_evidence:
            answer = "I don't have sufficient evidence in the provided documents to answer this question confidently."
            citations = []
            insufficient_evidence = True
        else:
            # Convert chunks to format expected by prompt_manager
            chunks_info = []
            for chunk in chunks:
                chunks_info.append(
                    {
                        "text": chunk["content"],
                        "filename": chunk["document_filename"],
                        "page": chunk["page_number"],
                    }
                )

            # Prepare context and generate answer
            context = prompt_manager.format_context(chunks_info)
            messages = prompt_manager.get_prompt(intent, question, context)

            # Generate answer
            raw_answer = mistral_client.chat_completion(
                messages=messages, temperature=0.1, max_tokens=1000
            )

            # Post-process answer
            answer, citations, insufficient_evidence = (
                answer_postprocessor.process_answer(raw_answer, chunks_info)
            )

        # Display answer
        console.print(Panel(answer, title="Generated Answer"))

        if citations:
            console.print(f"\n[dim]Citations found: {len(citations)} citations[/dim]")
            for i, citation in enumerate(citations[:3], 1):  # Show first 3
                console.print(
                    f"[dim]  {i}. {citation.filename} p.{citation.page}: {citation.snippet[:100]}...[/dim]"
                )
        else:
            console.print("\n[dim]No citations found in answer[/dim]")

        if insufficient_evidence:
            console.print("[yellow]Warning: Insufficient evidence detected[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def stats(db_path: str | None = typer.Option(None, "--db", help="Database path")):
    """Show database statistics."""
    db_path = db_path or config.DB_PATH

    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats_queries = [
        ("Total Documents", "SELECT COUNT(*) FROM documents"),
        ("Total Chunks", "SELECT COUNT(*) FROM chunks"),
        ("Total Embeddings", "SELECT COUNT(*) FROM embeddings"),
        ("Avg Chunk Length", "SELECT AVG(LENGTH(text)) FROM chunks"),
        ("Max Chunk Length", "SELECT MAX(LENGTH(text)) FROM chunks"),
        ("Min Chunk Length", "SELECT MIN(LENGTH(text)) FROM chunks"),
        ("Total Tokens (TF-IDF)", "SELECT COUNT(*) FROM tokens"),
        ("Unique Terms (TF-IDF)", "SELECT COUNT(DISTINCT token) FROM df"),
    ]

    table = Table(title="Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric, query in stats_queries:
        result = conn.execute(query).fetchone()[0]
        if "Avg" in metric or "Length" in metric:
            value = f"{result:.1f}" if result else "0"
        else:
            value = str(result) if result else "0"
        table.add_row(metric, value)

    console.print(table)

    # Document breakdown
    doc_query = """
    SELECT d.filename, COUNT(c.id) as chunk_count,
           AVG(LENGTH(c.text)) as avg_length
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.doc_id
    GROUP BY d.id, d.filename
    ORDER BY chunk_count DESC
    """

    cursor = conn.execute(doc_query)
    doc_rows = cursor.fetchall()

    if doc_rows:
        console.print("\n")
        doc_table = Table(title="Documents Breakdown")
        doc_table.add_column("Document", style="yellow")
        doc_table.add_column("Chunks", style="cyan")
        doc_table.add_column("Avg Length", style="green")

        for row in doc_rows:
            doc_table.add_row(
                row["filename"],
                str(row["chunk_count"]),
                f"{row['avg_length']:.1f}" if row["avg_length"] else "0",
            )

        console.print(doc_table)

    conn.close()


if __name__ == "__main__":
    app()
