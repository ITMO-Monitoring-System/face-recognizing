import os
from face_service.core.db_pg import DbConfig, load_persons_from_db
from face_service.core.recognize import recognize_bgr

if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Set DATABASE_URL env, e.g. postgresql://user:pass@host:5432/db")

    persons = load_persons_from_db(DbConfig(dsn=dsn))
    print("Loaded persons:", len(persons))


    print(recognize_image("/faces/lev/img.png"
                          , persons, threshold=0.45))
