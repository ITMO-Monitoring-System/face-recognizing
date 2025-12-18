from insightface.app import FaceAnalysis

app = FaceAnalysis(allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=-1, det_size=(640, 640))

print("Model downloaded successfully")
