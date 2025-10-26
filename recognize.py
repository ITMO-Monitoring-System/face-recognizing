import json, cv2, numpy as np
from retina_embending import get_face_embeddings

DB_PATH = "embeddings_db.json"

#нормализация(косинусовое сходство)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    persons = {}
    for row in raw:
        persons.setdefault(row["person_id"], []).append(np.array(row["embedding"], dtype=np.float32))
    return persons

def recognize_image(path: str, threshold: float = 0.45):
    img = cv2.imread(path)
    faces = get_face_embeddings(img)
    if not faces:
        return []
    persons = load_db()
    results = []
    for face in faces:
        emb = face["embedding"]
        best_id, best_score = None, -1.0
        for pid, embs in persons.items():
            score = max(cosine_sim(emb, e) for e in embs)
            if score > best_score:
                best_id, best_score = pid, score
        match = best_score >= threshold
        results.append({
            "bbox": face["bbox"],
            "person_id": best_id if match else None,
            "score": round(best_score, 4),
            "matched": match,
        })
    return results

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "probe.jpg"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    print(recognize_image(path, threshold=thr))
