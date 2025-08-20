"""CLI commands for database inspection and debug queries."""

import sqlite3
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.core.config import config
from app.logic.intent import ConversationMessage
from app.logic.intent import LLMIntentDetector
from app.logic.postprocess import HALLUCINATION_ERROR_MESSAGE
from app.logic.postprocess import PII_ERROR_MESSAGE
from app.services.query_service import RAGQueryService

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
        # Initialize the unified query service
        query_service = RAGQueryService(db_path=db_path)

        # Use the service to process the query
        result = query_service.query(
            query=question,
            conversation_history=[],  # CLI doesn't maintain conversation history
            top_k=k,
            rerank_k=k * 3,  # Get more candidates for reranking
            threshold=threshold,
            alpha=alpha,
            lambda_param=mmr_lambda,
            use_mmr=True,
        )

        # Handle service errors
        if not result.success:
            console.print(f"[red]Query service error: {result.error}[/red]")
            return
        if PII_ERROR_MESSAGE in result.answer:
            console.print("PII flagged")
            console.print(no_llm)
        if HALLUCINATION_ERROR_MESSAGE in result.answer:
            console.print("Hallucination flagged")

        # For CLI debugging, we'll show mock retrieval results based on citations
        # This provides similar information to the original implementation
        if result.citations:
            table = Table(title=f"Retrieved Chunks (k={k}, α={alpha}, λ={mmr_lambda})")
            table.add_column("Rank", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Document", style="yellow")
            table.add_column("Page", style="blue")
            table.add_column("Content Preview", style="white")

            for i, citation in enumerate(result.citations, 1):
                content_preview = (
                    citation.snippet[:CHUNK_PREVIEW_LENGTH] + "..."
                    if len(citation.snippet) > CHUNK_PREVIEW_LENGTH
                    else citation.snippet
                )

                # Use debug info for score if available, otherwise show estimated score
                score = (
                    result.debug_info.semantic_top1
                    if result.debug_info and i == 1
                    else threshold + 0.1
                )

                table.add_row(
                    str(i),
                    f"{score:.4f}",
                    citation.filename,
                    str(citation.page),
                    content_preview,
                )

            console.print(table)

        # Check evidence threshold using debug info
        best_score = result.debug_info.semantic_top1 if result.debug_info else 0.0
        has_evidence = not result.insufficient_evidence

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

        # Generate answer (already done by the service)
        console.print("\n" + "=" * 50)
        console.print("[bold]Answer generated using unified service...[/bold]")
        console.print(f"[dim]Detected intent: {result.intent}[/dim]")

        # Display answer
        console.print(Panel(result.answer, title="Generated Answer"))

        if result.citations:
            console.print(
                f"\n[dim]Citations found: {len(result.citations)} citations[/dim]"
            )
            for i, citation in enumerate(result.citations[:3], 1):  # Show first 3
                console.print(
                    f"[dim]  {i}. {citation.filename} p.{citation.page}: {citation.snippet[:100]}...[/dim]"
                )
        else:
            console.print("\n[dim]No citations found in answer[/dim]")

        if result.insufficient_evidence:
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


@app.command()
def test_intent(
    query: str = typer.Argument(..., help="Query to analyze for intent and rewriting"),
    history: list[str] = typer.Option(
        None,
        "--history",
        "-h",
        help="Conversation history (user:message or assistant:message format)",
    ),
    use_llm: bool = typer.Option(
        True, "--llm/--fallback", help="Use LLM-based detection or fallback only"
    ),
    show_details: bool = typer.Option(
        True, "--details/--simple", help="Show detailed analysis or simple output"
    ),
):
    """Test the LLM-based intent detection and query rewriting system."""
    console.print(
        Panel("🧪 [bold blue]Intent Detection & Query Rewriting Test[/bold blue]")
    )

    # Parse conversation history
    conversation_history = []
    if history:
        console.print(f"\n[dim]Processing {len(history)} history messages...[/dim]")
        for msg in history:
            if ":" not in msg:
                console.print(
                    f"[yellow]Warning: Invalid history format '{msg}'. Use 'user:message' or 'assistant:message'[/yellow]"
                )
                continue

            role, content = msg.split(":", 1)
            role = role.strip().lower()
            content = content.strip()

            if role not in ["user", "assistant"]:
                console.print(
                    f"[yellow]Warning: Invalid role '{role}'. Use 'user' or 'assistant'[/yellow]"
                )
                continue

            conversation_history.append(ConversationMessage(role=role, content=content))

    # Display conversation history if provided
    if conversation_history:
        console.print("\n[bold]Conversation History:[/bold]")
        for _i, msg in enumerate(conversation_history, 1):
            role_color = "blue" if msg.role == "user" else "green"
            console.print(
                f"  [{role_color}]{msg.role.title()}:[/{role_color}] {msg.content}"
            )

    console.print(f"\n[bold]Current Query:[/bold] [yellow]'{query}'[/yellow]")

    # Initialize detector
    try:
        detector = LLMIntentDetector()
        console.print("[dim]✅ LLMIntentDetector initialized[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize LLMIntentDetector: {e}[/red]")
        return

    # Test intent detection
    try:
        if use_llm and config.MISTRAL_API_KEY:
            console.print("[dim]🚀 Using LLM-based detection with Mistral AI...[/dim]")
            result = detector.detect_intent_and_rewrite(query, conversation_history)
            method_used = "LLM (Mistral AI)"
        else:
            if not config.MISTRAL_API_KEY:
                console.print(
                    "[yellow]⚠️ MISTRAL_API_KEY not set, using fallback detection[/yellow]"
                )
            else:
                console.print("[dim]🔧 Using fallback detection as requested[/dim]")
            result = detector._fallback_intent_detection(query)
            method_used = "Fallback (Pattern Matching)"

    except Exception as e:
        console.print(f"[red]❌ Error during intent detection: {e}[/red]")
        console.print("[yellow]Falling back to pattern matching...[/yellow]")
        result = detector._fallback_intent_detection(query)
        method_used = "Fallback (Error Recovery)"

    # Display results
    if show_details:
        # Create results table
        results_table = Table(
            title="Intent Detection Results",
            show_header=True,
            header_style="bold magenta",
        )
        results_table.add_column("Attribute", style="cyan", width=20)
        results_table.add_column("Value", style="white", width=60)

        results_table.add_row("Method Used", f"[green]{method_used}[/green]")
        results_table.add_row(
            "Detected Intent", f"[bold yellow]{result.intent}[/bold yellow]"
        )
        results_table.add_row("Confidence", f"[green]{result.confidence:.2f}[/green]")
        results_table.add_row("Original Query", f"[dim]{query}[/dim]")
        results_table.add_row(
            "Rewritten Query", f"[bold]{result.rewritten_query}[/bold]"
        )
        results_table.add_row("Reasoning", f"[dim]{result.reasoning}[/dim]")

        console.print(results_table)

        # Show response template
        template = detector.get_response_template(result.intent)
        console.print("\n[bold]Response Template:[/bold]")
        console.print(Panel(template, border_style="dim"))

    else:
        # Simple output
        console.print(
            f"\n[green]✅ Intent:[/green] [bold]{result.intent}[/bold] ({result.confidence:.2f})"
        )
        console.print(
            f"[green]✅ Rewritten:[/green] [bold]{result.rewritten_query}[/bold]"
        )

    # Show API status
    console.print(
        f"\n[dim]API Status: {'✅ Connected' if config.MISTRAL_API_KEY else '❌ No API Key'}[/dim]"
    )


if __name__ == "__main__":
    app()
