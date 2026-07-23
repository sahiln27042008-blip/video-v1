"""Command-line interface for renderer (supports both old and new formats)."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .renderer import Renderer
from .models import RenderConfig

app = typer.Typer(name="renderer", help="Render final video from editor plan.", add_completion=False)
console = Console()

@app.command()
def render(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to original video file"),
    editor_plan: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to editor_plan.json or technical_edit_plan.json"),
    candidate_clips: Optional[Path] = typer.Option(None, "--candidates", help="Path to candidate_clips.json (required for legacy plan)"),
    timeline_words: Optional[Path] = typer.Option(None, "--timeline-words", help="Path to timeline_with_words.json (required for legacy plan)"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    background_music: Optional[Path] = typer.Option(None, "--music", help="Path to background music file"),
    temp_dir: Path = typer.Option(Path("./temp_render"), "--temp", help="Temporary directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Render final video from editor plan (supports both old and new formats)."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        config = RenderConfig(
            video_path=str(video),
            editor_plan_path=str(editor_plan),
            candidate_clips_path=str(candidate_clips) if candidate_clips else None,
            timeline_words_path=str(timeline_words) if timeline_words else None,
            output_dir=str(output_dir),
            background_music_path=str(background_music) if background_music else None,
            temp_dir=str(temp_dir),
        )

        renderer = Renderer(config)
        console.print(f"[bold]Rendering video from:[/] {video}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Rendering...", total=None)
            result = renderer.render()
            progress.update(task, completed=True)

        if result.success:
            console.print(f"[OK] Final video written to: [bold]{result.output_video}[/]")
            if result.output_subtitles:
                console.print(f"[OK] Subtitles written to: [bold]{result.output_subtitles}[/]")
            console.print("[bold green]✅ Rendering completed successfully![/bold green]")
        else:
            console.print(f"[red]❌ Rendering failed: {result.error}[/red]")
            for log in result.log:
                console.print(log)
            raise typer.Exit(code=1)

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()