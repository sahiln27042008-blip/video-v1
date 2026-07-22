"""Command-line interface for conversation layer (WhisperX)."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .processor import ConversationProcessor

app = typer.Typer(name="conversation-layer", help="Transcribe and diarize with WhisperX.", add_completion=False)
console = Console()


@app.command()
def process(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to input video"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file path."),
    model_size: str = typer.Option("base", "--model", help="Whisper model size (tiny, base, small, medium, large)."),
    device: str = typer.Option("cpu", "--device", help="Device (cpu, cuda)."),
    compute_type: str = typer.Option("int8", "--compute-type", help="Compute type for whisper."),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", help="Hugging Face token for diarization."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output."),
) -> None:
    """Run conversation processing and save JSON."""
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        processor = ConversationProcessor(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            hf_token=hf_token,
        )

        console.print(f"[bold]Processing conversation in:[/] {video}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Transcribing and diarizing...", total=None)
            result = processor.process(
                str(video),
                str(output) if output else None,
            )
            progress.update(task, completed=True)

        if not output:
            console.print(result.model_dump_json(indent=2))
        else:
            console.print(f"[green]✓[/] Output written to: [bold]{output}[/]")

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()