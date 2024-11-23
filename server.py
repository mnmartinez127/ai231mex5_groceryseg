import io
import os
import argparse
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from fastapi import FastAPI, File, UploadFile, Request
    from fastapi.responses import StreamingResponse,JSONResponse
    from fastapi.encoders import jsonable_encoder
except ImportError:
    os.system("pip install opencv-python opencv-python-contrib")
    os.system("pip install 'numpy<2.0'")
    os.system("pip install ultralytics")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install fastapi starlette")
    #Best to install these manually
    #os.system("pip install onnx onnxslim onnxruntime-gpu")
    #os.system("pip install tensorrt")
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from fastapi import FastAPI, File, UploadFile, Request
    from fastapi.responses import StreamingResponse,JSONResponse
    from fastapi.encoders import jsonable_encoder
#Define initial constants, should not change!
SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
#allows for custom port setting
PORT_NUMBER = 8002
parser = argparse.ArgumentParser(description="Specify the port used by the server",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p",default=PORT_NUMBER,type=int,help="Server port")
args = parser.parse_args()
PORT_NUMBER = args.p
#If anyone says your client is using someone else's server after you run this, show this line
if os.system(f"ss -ltnup | grep ':{PORT_NUMBER}' | grep -o 'pid=[0-9]*' | grep -o '[0-9]*'") == 0:
    print(f"Another process is using the current port. Terminating...")
    os.system(f"kill -9 `ss -ltnup | grep ':{PORT_NUMBER}' | grep -o 'pid=[0-9]*' | grep -o '[0-9]*'`")


DEFAULT_MODEL = "YOLOv11 Segmentation M"
DEFAULT_MODE = "torch"
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LIST = list(range(torch.cuda.device_count()))
MODE_LIST = ["torch","onnx"]
#Build dict of available models
MODELS_DIR = os.path.join(os.getcwd(), "models")
model_dict = {}
for model_dir in os.listdir(MODELS_DIR):
    weight_path = os.path.join(MODELS_DIR, model_dir, "weights", "best.pt")
    if os.path.exists(weight_path):
        model_dict[model_dir] = weight_path
MODEL_LIST = sorted(list(model_dict.keys()))


#Initialize session parameters
params = {
    "model_name":DEFAULT_MODEL,
    "model_mode":DEFAULT_MODE,
    "model_device":DEFAULT_DEVICE,
    "conf":0.5,
    "iou":0.1,
    "classes":[],
    "show_boxes":True,
    "show_masks":True,
}

server_lists = {
    "models":MODEL_LIST,
    "devices":DEVICE_LIST,
    "modes":MODE_LIST
}

#The default parameter is persistent. This is intended and will be taken advantage of.
def load_model(model_name="",model_mode="torch",model_device="cpu",model=[None]):
    torch.cuda.empty_cache()
    model_path = model_dict.get(model_name,"")
    if not os.path.isfile(model_path):
        model[0] = None
    else:
        match model_mode:
            case "onnx":
                if os.path.exists(os.path.splitext(model_path)[0]+".onnx"):
                    model[0] = YOLO(os.path.splitext(model_path)[0]+".onnx",task="segment")
                else:
                    model[0] = YOLO(model_path,task="segment")
                    new_model_path = model[0].export(format="onnx",device=model_device)
                    model[0] = YOLO(new_model_path,task="segment")
            #case "tensorrt":
            #    if os.path.exists(os.path.splitext(model_path)[0]+".engine"):
            #        model[0] = YOLO(os.path.splitext(model_path)[0]+".engine",task="segment")
            #    else:
            #        model[0] = YOLO(model_path,task="segment")
            #        new_model_path = model[0].export(format="engine",device=model_device)
            #        model[0] = YOLO(new_model_path,task="segment")
            case _:
                    model[0] = YOLO(model_path,task="segment")
    torch.cuda.empty_cache()
    return model

#Use a list as a pointer to a Model
model:list[YOLO|None] = load_model(DEFAULT_MODEL,DEFAULT_MODE,DEFAULT_DEVICE) # Load the default model

app = FastAPI() #start the server

#send the model, device, and mode list to the client
@app.get("/get_lists")
async def get_lists():
    return JSONResponse(content=jsonable_encoder(server_lists))

@app.post("/infer_frame")
async def infer_frame(request:Request,file: UploadFile = File(...)):
    image_bytes = await file.read() #read the sent image
    new_params = request.query_params #read the sent parameters. if they exist
    #Update the parameters using the new ones
    if new_params["model_name"] != params["model_name"] or new_params["model_mode"] != params["model_mode"]:
        load_model(new_params["model_name"],new_params["model_mode"],new_params["model_device"],model)
    #Load the parameters correctly
    params["model_name"] = new_params["model_name"]
    params["model_mode"] = new_params["model_mode"]
    devices = new_params.getlist("classes")
    if len(devices) == 1:
        params["model_device"] = int(devices[0]) if devices[0] != "cpu" else "cpu"
    else:
        params["model_device"] = [int(i) for i in devices]
    params["conf"] = float(new_params["conf"])
    params["iou"] = float(new_params["iou"])
    params["classes"] = None if not new_params.getlist("classes") else [int(i) for i in new_params.getlist("classes")]
    params["show_boxes"] = new_params["show_boxes"].lower() == "true"
    params["show_masks"] = new_params["show_masks"].lower() == "true"

    #Perform inference on the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if model[0] is None:
        _,buffer = cv2.imencode('.jpg', image)
    else:
        results = model[0].track(image,
                                    conf = params["conf"],
                                    iou = params["iou"],
                                    classes = params["classes"],
                                    device = params["model_device"],
                                    )
        result_frame =  results[0].plot(boxes = params["show_boxes"],
                                            masks = params["show_masks"]
                                            )
        _, buffer = cv2.imencode('.jpg', result_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER)