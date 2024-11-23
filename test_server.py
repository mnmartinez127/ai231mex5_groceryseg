import io
import os
import sys
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from fastapi import FastAPI, File, UploadFile
    from starlette.responses import StreamingResponse
except ImportError:
    os.system("pip install opencv-python opencv-python-contrib")
    os.system("pip install 'numpy<2.0'")
    os.system("pip install ultralytics")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install fastapi starlette")
    #Best to install these manually
    #os.system("pip install onnx onnxslim onnxruntime-gpu")
    #os.system("pip install tensorrt")


#Define initial constants, should not change!
SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
#allows for custom port setting
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8003
PAGE = "/infer_frame"
SERVER_URL = f"{SERVER_ADDRESS}:{PORT_NUMBER}{PAGE}"
DEFAULT_MODEL = "YOLOv11 Segmentation M"
DEFAULT_MODE = "torch"
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LIST = list(range(torch.cuda.device_count()))
MODE_LIST = ["torch","onnx","tensorrt"]

#Initialize session parameters
params = {
    "model_name":"No Model",
    "model_mode":"torch",
    "model_device":DEFAULT_DEVICE,
    "server_url":SERVER_URL,
    "conf":0.5,
    "iou":0.1,
    "classes":[],
    "show_boxes":True,
    "show_masks":True,
}



MODEL_PATH = os.path.join(os.getcwd(),"models",DEFAULT_MODEL,"weights","best.pt")
model = YOLO(MODEL_PATH)  # Load your trained segmentation model
app = FastAPI()

@app.post("/infer_frame")
async def infer_frame(file: UploadFile = File(...)):
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform inference
    results = model(frame,conf=params["conf"],iou=params["iou"],classes=params["classes"],device=params["model_device"])

    # Check if `results` is a list, take the first element if so
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    # Render the segmentation result on the frame
    segmented_frame = result.plot(boxes=params["show_boxes"],masks=params["show_masks"])  # `plot()` is the method to overlay results on the image


    # Encode the segmented frame for sending back
    _, buffer = cv2.imencode('.jpg', segmented_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER)