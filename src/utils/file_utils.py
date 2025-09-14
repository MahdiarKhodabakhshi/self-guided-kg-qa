"""
File utility functions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def save_json(data: List[Dict[str, Any]], path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
