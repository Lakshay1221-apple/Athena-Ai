"""Common utility functions for Athena AI."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Union


def ensure_directory(path: Union[Path, str]) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def calculate_text_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def calculate_file_hash(path: Union[Path, str]) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_json(path: Union[Path, str]) -> Any:
    """Load JSON from a file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Union[Path, str], indent: int = 2) -> Path:
    """Save data as JSON to a file with directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return p


def load_jsonl(path: Union[Path, str]) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    p = Path(path)
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Union[Path, str], mode: str = "w") -> Path:
    """Save records to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def append_jsonl(record: Dict[str, Any], path: Union[Path, str]) -> Path:
    """Append a single record to a JSONL file."""
    return save_jsonl([record], path, mode="a")
