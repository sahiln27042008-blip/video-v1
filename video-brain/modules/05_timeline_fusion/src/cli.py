"""Command-line interface for timeline fusion."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .fuser import TimelineFuser

app = typer.Typer(name="timeline-fusion", help="Fuse all data into a single timeline.", add_completion=False)
console = Console()


@app.command()
def fuse(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input video"),
    scenes_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to scenes.json"),
    faces_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to faces.json"),
    identities_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to identities.json"),
    conversation_json: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to conversation.json"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path (timeline.json)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Run timeline fusion and save JSON. Also generates people.json in the same directory."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        fuser = TimelineFuser()
        console.print(f"[bold]Fusing timeline from:[/] {video}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Fusing data...", total=None)
            result, people = fuser.process(
                str(video),
                str(scenes_json),
                str(faces_json),
                str(identities_json),
                str(conversation_json),
                str(output) if output else None,
            )
            progress.update(task, completed=True)

        if output:
            console.print(f"[OK] Timeline written to: [bold]{output}[/]")
            # people.json is written by fuser.process
            people_path = output.parent / "people.json"
            if people_path.exists():
                console.print(f"[OK] People mapping written to: [bold]{people_path}[/]")
        else:
            console.print(result.model_dump_json(indent=2))

        console.print(f"\n[bold]Summary:[/] {len(result.segments)} timeline segments, {len(people)} persons")

    except Exception as e:
        sys.stderr.write(f"[ERROR] {e}\n")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()