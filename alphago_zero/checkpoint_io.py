"""Atomic local artifacts; interrupted writes preserve the last complete file."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import IO, Iterator


@contextmanager
def atomic_output(path: Path, *, keep_previous: bool = False) -> Iterator[IO[bytes]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    backup_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream.file
            stream.flush()
            os.fsync(stream.fileno())
        if keep_previous and path.exists():
            previous = path.with_name(f"{path.stem}.previous{path.suffix}")
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{previous.name}.", suffix=".tmp", delete=False) as backup:
                backup_temporary = Path(backup.name)
                with path.open("rb") as source:
                    shutil.copyfileobj(source, backup)
                backup.flush()
                os.fsync(backup.fileno())
            os.replace(backup_temporary, previous)
        os.replace(temporary, path)
    finally:
        for candidate in (temporary, backup_temporary):
            if candidate is not None:
                candidate.unlink(missing_ok=True)
