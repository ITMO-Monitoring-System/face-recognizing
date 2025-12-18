import json, cv2
from pathlib import Path
from face_service.core.retina_embending import get_face_embeddings

DATA_DIR = Path("../faces")
DB_PATH = Path("../embeddings_db.json")

def enroll_folder():
    db = []
    for person_dir in DATA_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name
        for img_path in person_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = get_face_embeddings(img)
            if not faces:
                continue
            #тут выбирается самое сигма лицо
            best = max(faces, key=lambda f: f["det_score"])
            emb = best["embedding"].tolist()
            db.append({"person_id": person_id, "embedding": emb, "src": str(img_path)})
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    print(f"Сохранено: {len(db)} эмбеддингов → {DB_PATH}")

if __name__ == "__main__":
    enroll_folder()