1 - убей себя  
2 - рожон   
3 - гумно   

python -m venv .venv    
.\.venv\Scripts\activate   
python -m pip install -U pip setuptools wheel

pip install --no-cache-dir numpy==1.26.4   
pip install --no-cache-dir onnxruntime==1.18.0   
pip install --no-cache-dir opencv-python==4.10.0.84   
pip install --no-cache-dir --no-deps insightface==0.7.3   
pip install --no-cache-dir --no-deps albucore==0.0.24   
pip install --no-cache-dir --no-deps albumentations==1.4.17   
pip install --no-cache-dir fastapi uvicorn


ПРОВЕРКА

import numpy, cv2, onnxruntime, insightface   
print("numpy", numpy.__version__)   
print("opencv", cv2.__version__)   
print("onnxruntime", onnxruntime.__version__)   
print("insightface", insightface.__version__)   

