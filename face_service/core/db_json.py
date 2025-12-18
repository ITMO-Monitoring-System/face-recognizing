import json
from pathlib import Path
import numpy as np

def load_persons_from_json(db_path: str | Path) -> dict[str, list[np.ndarray]]:
    db_path = Path(db_path)
    with open(db_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    persons: dict[str, list[np.ndarray]] = {}
    for row in raw:
        pid = row["person_id"]
        emb = np.array(row["embedding"], dtype=np.float32)
        persons.setdefault(pid, []).append(emb)
    return persons
