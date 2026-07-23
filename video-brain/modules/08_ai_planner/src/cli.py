"""Command-line interface for AI planner."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .planner import Planner
from .providers import DeepSeekProvider, OpenAIAProvider

app = typer.Typer(name="ai-planner", help="AI-powered clip selection.", add_completion=False)
console = Console()


@app.command()
def plan(
    candidate_clips: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to candidate_clips.json"),
    timeline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to timeline_with_words.json"),
    people: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to people.json"),
    metrics: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to metrics.json"),
    segment_metrics: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to segment_metrics.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    provider: str = typer.Option("deepseek", "--provider", help="LLM provider: deepseek, openai"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name (overrides default)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or set env var)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Run AI planning and save final clip plan."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        # Select provider
        if provider.lower() == "deepseek":
            provider_instance = DeepSeekProvider(api_key=api_key, model=model or "deepseek-chat")
        elif provider.lower() == "openai":
            provider_instance = OpenAIAProvider(api_key=api_key, model=model or "gpt-4")
        else:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
            raise typer.Exit(code=1)

        planner = Planner(provider=provider_instance)

        console.print(f"[bold]Planning with provider:[/] {provider}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Planning...", total=None)
            result = planner.plan(
                str(candidate_clips),
                str(timeline),
                str(people),
                str(metrics),
                str(segment_metrics),
                video_path=str(timeline.parent / "video.mp4") if timeline.parent else None,
            )
            progress.update(task, completed=True)

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            console.print(f"[OK] Output written to: [bold]{output}[/]")
        else:
            console.print(result.model_dump_json(indent=2))

        # Summary
        table = Table(title=f"Selected {len(result.selected_clips)} clips")
        table.add_column("Clip ID", style="dim")
        table.add_column("Score", width=8)
        table.add_column("Reason")
        for c in result.selected_clips:
            table.add_row(str(c.clip_id), f"{c.score:.1f}", c.reason[:50] + "...")
        console.print(table)

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()