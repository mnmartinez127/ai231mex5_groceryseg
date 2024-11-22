# inference server using fastapi (serve on DGX)
import os
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
import numpy as np
import torch
import io

import sys
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8003

DEVICE_LIST = list(range(torch.cuda.device_count()))
device = 0 if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.path.join(os.getcwd(),"models","YOLOv11 Segmentation M","weights","best.pt")
app = FastAPI()
model = YOLO(MODEL_PATH)  # Load your trained segmentation model

@app.post("/infer_frame")
async def infer_frame(file: UploadFile = File(...)):
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform inference
    results = model(frame)

    # Check if `results` is a list, take the first element if so
    if isinstance(results, list):
        result = results[0]
    else:
        result = results

    # Render the segmentation result on the frame
    segmented_frame = result.plot()  # `plot()` is the method to overlay results on the image


    # Encode the segmented frame for sending back
    _, buffer = cv2.imencode('.jpg', segmented_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER)