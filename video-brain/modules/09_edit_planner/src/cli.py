"""Command-line interface for AI edit planner."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .planner import EditPlanner
from .providers import DeepSeekProvider, OpenAIAProvider

app = typer.Typer(name="edit-planner", help="AI-powered editing instructions.", add_completion=False)
console = Console()


@app.command()
def plan(
    final_clip_plan: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to final_clip_plan.json"),
    timeline: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to timeline_with_words.json"),
    people: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to people.json"),
    metrics: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to metrics.json"),
    segment_metrics: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to segment_metrics.json"),
    candidate_clips: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to candidate_clips.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    provider: str = typer.Option("deepseek", "--provider", help="LLM provider: deepseek, openai"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name (overrides default)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key (or set env var)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Run AI edit planning and save edit_plan.json."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        if provider.lower() == "deepseek":
            provider_instance = DeepSeekProvider(api_key=api_key, model=model or "deepseek-chat")
        elif provider.lower() == "openai":
            provider_instance = OpenAIAProvider(api_key=api_key, model=model or "gpt-4")
        else:
            console.print(f"[red]Unsupported provider: {provider}[/red]")
            raise typer.Exit(code=1)

        planner = EditPlanner(provider=provider_instance)

        console.print(f"[bold]Planning with provider:[/] {provider}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Planning edits...", total=None)
            result = planner.plan(
                str(final_clip_plan),
                str(timeline),
                str(people),
                str(metrics),
                str(segment_metrics),
                str(candidate_clips),
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
        table = Table(title=f"Edit plan for {len(result.clips)} clips")
        table.add_column("Clip ID", style="dim")
        table.add_column("Style")
        table.add_column("Music")
        table.add_column("Confidence")
        for c in result.clips:
            table.add_row(str(c.clip_id), c.editing_style, c.music, f"{c.confidence:.2f}")
        console.print(table)

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()