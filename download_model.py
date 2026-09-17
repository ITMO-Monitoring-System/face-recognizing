"""Download, validate and warm the exact models used by the service."""
from face_service.core.retina_embending import get_app

if __name__ == "__main__":
    get_app()
    print("Model ready")
