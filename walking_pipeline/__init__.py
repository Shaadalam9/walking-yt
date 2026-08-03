"""YouTube pedestrian walking video research pipeline."""


def main() -> None:
    """Load and run the pipeline without importing heavy models eagerly."""
    from .pipeline import main as run_pipeline

    run_pipeline()


__all__ = ["main"]
