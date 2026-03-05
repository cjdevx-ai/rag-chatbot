"""
main.py
───────
CLI chatbot entry point. Uses `rich` for a polished terminal experience.

Commands available during chat:
  /quit or /exit   → exit the chatbot
  /reset           → clear conversation history
  /sources         → show the retrieved sources for the last query
  /docs            → list all indexed documents
  /help            → show commands
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from rag.pipeline import RAGPipeline

load_dotenv()

console = Console()

# ── helpers ───────────────────────────────────────────────────────────────────

def print_header():
    console.print(Panel(
        Text.assemble(
            ("  ██████╗  █████╗  ██████╗ \n", "bold cyan"),
            ("  ██╔══██╗██╔══██╗██╔════╝ \n", "bold cyan"),
            ("  ██████╔╝███████║██║  ███╗\n", "bold blue"),
            ("  ██╔══██╗██╔══██║██║   ██║\n", "bold blue"),
            ("  ██║  ██║██║  ██║╚██████╔╝\n", "bold magenta"),
            ("  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ \n\n", "bold magenta"),
            ("  Retrieval-Augmented Generation CLI Chatbot\n", "bold white"),
            ("  Type /help for commands", "dim"),
        ),
        border_style="cyan",
        padding=(1, 2),
    ))


def print_help():
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Command", style="cyan bold")
    table.add_column("Description", style="white")
    for cmd, desc in [
        ("/quit, /exit", "Exit the chatbot"),
        ("/reset",       "Clear conversation history"),
        ("/sources",     "Show sources from the last query"),
        ("/docs",        "List all indexed document chunks"),
        ("/help",        "Show this help message"),
    ]:
        table.add_row(cmd, desc)
    console.print(Panel(table, title="[bold]Commands[/bold]", border_style="dim"))


def print_sources(retrieved):
    if not retrieved:
        console.print("[dim]No sources retrieved for this query.[/dim]")
        return
    table = Table(
        title="Retrieved Context",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", width=3)
    table.add_column("Source", style="green")
    table.add_column("Chunk", width=5)
    table.add_column("Score", width=6)
    table.add_column("Preview", style="dim")
    for i, (chunk, score) in enumerate(retrieved, 1):
        preview = chunk.text[:80].replace("\n", " ") + "…"
        table.add_row(
            str(i),
            chunk.source,
            str(chunk.chunk_id),
            f"{score:.3f}",
            preview,
        )
    console.print(table)


def print_answer(answer: str):
    console.print(Panel(
        Markdown(answer),
        title="[bold green]Assistant[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print_header()

    # Config from env
    api_key    = os.getenv("GEMINI_API_KEY")
    embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    top_k       = int(os.getenv("TOP_K", "3"))
    chunk_size  = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    docs_dir    = "docs"

    if not api_key:
        console.print("[bold red]ERROR:[/bold red] GEMINI_API_KEY not set. "
                      "Create a .env file from .env.example.")
        sys.exit(1)

    if not Path(docs_dir).is_dir():
        console.print(f"[bold red]ERROR:[/bold red] Docs directory '{docs_dir}' not found.")
        sys.exit(1)

    # Build pipeline
    console.print("\n[bold cyan]Initialising RAG pipeline...[/bold cyan]")
    try:
        pipeline = RAGPipeline(
            docs_dir=docs_dir,
            embed_model=embed_model,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            api_key=api_key,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialise pipeline:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[bold green]✓ Ready![/bold green] "
                  f"Indexed [cyan]{pipeline.vector_store.total}[/cyan] chunks from "
                  f"[cyan]{docs_dir}/[/cyan]\n")

    last_retrieved = []

    # ── chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        elif user_input.lower() == "/reset":
            pipeline.reset_history()
            last_retrieved = []
            continue
        elif user_input.lower() == "/sources":
            print_sources(last_retrieved)
            continue
        elif user_input.lower() == "/docs":
            docs = {}
            for chunk in pipeline.chunks:
                docs.setdefault(chunk.source, 0)
                docs[chunk.source] += 1
            table = Table(title="Indexed Documents", box=box.ROUNDED, border_style="cyan")
            table.add_column("File", style="green")
            table.add_column("Chunks", style="cyan")
            for src, count in sorted(docs.items()):
                table.add_row(src, str(count))
            console.print(table)
            continue
        elif user_input.lower() == "/help":
            print_help()
            continue

        # RAG query
        with console.status("[bold cyan]Retrieving & generating...[/bold cyan]"):
            try:
                answer, last_retrieved = pipeline.query(user_input)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
                continue

        print_answer(answer)

        # Show source names inline
        if last_retrieved:
            sources = ", ".join(
                f"[cyan]{c.source}[/cyan] ({s:.2f})"
                for c, s in last_retrieved
            )
            console.print(f"  [dim]Sources: {sources}[/dim]\n")


if __name__ == "__main__":
    main()
