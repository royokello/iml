from __future__ import annotations

import shutil
from pathlib import Path


def copy_files(source_dir: str | Path, target_dir: str | Path, names: tuple[str, ...]) -> list[Path]:
    """Copy a fixed list of files from one folder to another.

    A file that already exists in the target with the same size is skipped.
    Nothing happens when the source and the target are the same folder.

    Args:
        source_dir: The folder that holds the files.
        target_dir: The folder to copy into. It is created if it is missing.
        names: The file names to copy.

    Returns:
        The paths in the target folder, in the order of `names`.

    Raises:
        FileNotFoundError: If the source folder or a listed file is missing.
            No file is copied in that case.
    """
    source = Path(source_dir).expanduser().resolve()
    target = Path(target_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Files not found in {source}: {', '.join(missing)}")

    if source == target:
        print(f"Source and target are the same folder: {source}. Nothing to copy.")
        return [target / name for name in names]

    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in names:
        source_path = source / name
        target_path = target / name
        if target_path.is_file() and target_path.stat().st_size == source_path.stat().st_size:
            print(f" * skip {name} (already there)")
        else:
            print(f" * copy {name}")
            shutil.copyfile(source_path, target_path)
        copied.append(target_path)
    return copied
