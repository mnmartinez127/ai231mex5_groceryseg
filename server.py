import io
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import StreamingResponse
models_dir = os.path.join(os.path.join(os.getcwd()), "models")
model_dict = {}
#get the last model from training
for model_dir in os.listdir(models_dir):
    weight_path = os.path.join(models_dir, model_dir, "weights", "last.pt")
    if os.path.exists(weight_path):
        model_dict[model_dir] = weight_path
model_name = sorted(list(model_dict.keys()))[0]
model = [YOLO(model_dict[model_name],task="segment")]  # Load your trained segmentation model
torch.cuda.empty_cache()

app = FastAPI()

@app.post("/infer_frame")
async def infer_frame(request:Request,file: UploadFile = File(...)):
    image_bytes = await file.read()
    params = request.query_params
    if params["model_name"] != model_name and params["model_name"] in model_dict.keys():
        model[0] = YOLO(model_dict[params["model_name"]],task="segment")
        torch.cuda.empty_cache()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    logger.debug(params)
    results = model[0].track(frame,conf=float(params["conf"]),iou=float(params["iou"]),classes=None if not params.getlist("classes") else [int(i) for i in params.getlist("classes")])
    segmented_frame = results[0].plot(boxes=(params["show_boxes"].lower() == "true"),masks=(params["show_masks"].lower() == "true"))
    _, buffer = cv2.imencode('.jpg', segmented_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8086)