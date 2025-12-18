
python -m venv .venv    
.\.venv\Scripts\activate   
python -m pip install -U pip setuptools wheel

pip install numpy==1.26.4   
pip install onnxruntime==1.18.0   
pip install opencv-python==4.10.0.84   
pip install insightface==0.7.3   
pip install  albucore==0.0.24   
pip install albumentations==1.4.17   
pip install fastapi uvicorn


ПРОВЕРКА

import numpy, cv2, onnxruntime, insightface   
print("numpy", numpy.__version__)   
print("opencv", cv2.__version__)   
print("onnxruntime", onnxruntime.__version__)   
print("insightface", insightface.__version__)   

