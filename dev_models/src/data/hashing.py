"""File checksums recorded in result files."""
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
