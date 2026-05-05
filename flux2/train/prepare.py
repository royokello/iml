from __future__ import annotations

from pathlib import Path


def precheck_project_dir(project_dir, resume):
    project_path = Path(project_dir)

    for image_dir in (project_path / "images" / "low", project_path / "images" / "high"):
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Project directory is missing: {image_dir}")

    if not resume:
        log_file = project_path / "models" / "steps.csv"
        if log_file.is_file():
            raise FileExistsError(f"Log file already exists: {log_file}")


def prepare_project_dir(project_dir):
    project_path = Path(project_dir)
    
    run_dir = project_path / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
