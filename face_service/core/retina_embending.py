import cv2
import numpy as np
from insightface.app import FaceAnalysis

#ctx_id = -1 - CPU
#ctx_id = 0 - GPU

app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

#лицо в эмбендинг
def get_face_embeddings(img_bgr: np.ndarray):
    faces = app.get(img_bgr)              # детект + выравнивание внутри
    out = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)#нормирую вектор
        bbox = f.bbox.astype(int).tolist()
        kps  = f.kps.astype(float).tolist()#ключевые точки на лице
        out.append({"embedding": emb, "bbox": bbox, "kps": kps, "det_score": float(f.det_score)})
    return out

if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    faces = get_face_embeddings(img)
    print(f"Найдено лиц: {len(faces)}")
    if faces:
        print("Размер эмбеддинга:", faces[0]["embedding"].shape)