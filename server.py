import io
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, Request
from starlette.responses import StreamingResponse
import sys
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8002

DEVICE_LIST = list(range(torch.cuda.device_count()))
model_device = 0 if torch.cuda.is_available() else "cpu"


models_dir = os.path.join(os.path.join(os.getcwd()), "models")
model_dict = {}
#get the last model from training
for model_dir in os.listdir(models_dir):
    weight_path = os.path.join(models_dir, model_dir, "weights", "last.pt")
    if os.path.exists(weight_path):
        model_dict[model_dir] = weight_path
model_name = sorted(list(model_dict.keys()))[0]
model_mode = "torch"
model = [YOLO(model_dict[model_name],task="segment")]  # Load your trained segmentation model
torch.cuda.empty_cache()

app = FastAPI()

def load_model(model_path,mode="torch"):
    torch.cuda.empty_cache()
    model = YOLO(model_path,task="segment")
    match mode:
        case "onnx":
            if os.path.exists(os.path.splitext(model_path)[0]+".onnx"):
                model = YOLO(os.path.splitext(model_path)[0]+".onnx",task="segment")
            else:
                new_model_path = model.export(format="onnx",device=model_device)
                model = YOLO(new_model_path,task="segment")
        case "tensorrt":
            if os.path.exists(os.path.splitext(model_path)[0]+".engine"):
                model = YOLO(os.path.splitext(model_path)[0]+".engine",task="segment")
            else:
                new_model_path = model.export(format="engine",device=model_device)
                model = YOLO(new_model_path,task="segment")
        case _:
            pass
    categories = [] if model.names is None else sorted(model.names.items(), key = lambda x:x[0])
    return model,categories




@app.post("/infer_frame")
async def infer_frame(request:Request,file: UploadFile = File(...)):
    image_bytes = await file.read()
    params = request.query_params
    if (params["model_name"] != model_name or params.get("mode","torch") != model_mode) and params["model_name"] in model_dict.keys():
        model[0],_ = load_model(model_dict[params["model_name"]],params.get("mode","torch"))
        model_name = params["model_name"]
        model_mode = params.get("mode","torch")
        torch.cuda.empty_cache()
    if "device" in params.keys():
        devices = params.getlist("classes")
        if len(devices) == 1:
            model_device = int(devices[0]) if devices[0] != "cpu" else "cpu"
        else:
            model_device = [int(i) for i in devices]
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model[0].track(frame,conf=float(params["conf"]),iou=float(params["iou"]),classes=None if not params.getlist("classes") else [int(i) for i in params.getlist("classes")])
    segmented_frame = results[0].plot(boxes=(params["show_boxes"].lower() == "true"),masks=(params["show_masks"].lower() == "true"))
    
    _, buffer = cv2.imencode('.jpg', segmented_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER)