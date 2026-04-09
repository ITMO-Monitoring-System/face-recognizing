import os

from insightface.app import FaceAnalysis

ctx_id = int(os.getenv("FACE_ANALYSIS_CTX_ID", "-1"))
det_size = int(os.getenv("FACE_ANALYSIS_DET_SIZE", "640"))

app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

print("Model downloaded successfully")
