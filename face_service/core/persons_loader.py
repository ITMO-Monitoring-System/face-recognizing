from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

from .db_json import load_persons_from_json
from .db_pg import DbConfig, load_persons_from_pg_embeddings

@dataclass(frozen=True)
class PersonsSource:
    mode: str  # "json" | "pg"
    json_path: Path = Path("embeddings_db.json")

def load_persons(source: PersonsSource) -> dict[str, list[np.ndarray]]:
    if source.mode == "json":
        return load_persons_from_json(source.json_path)

    if source.mode == "pg":
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is not set")
        return load_persons_from_pg_embeddings(DbConfig(dsn=dsn))

    raise ValueError(f"Unknown mode: {source.mode}")
