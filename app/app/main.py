import base64
import argparse
import time
import logging
import asyncio
import contextlib
import os
import io
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from websockets.exceptions import ConnectionClosed
from fastapi import (FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, Header)
from fastapi.responses import HTMLResponse,StreamingResponse,JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class Message(BaseModel):
    message: str

def setup_logger():
    FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    print(f'Created logger with name {__name__}')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(FORMAT)
    logger.addHandler(ch)
    return logger
logger = setup_logger()



SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
PORT_NUMBER = 8002 #inference server
#PORT_NUMBER = 8003 #test server

#Parse terminal arguments
parser = argparse.ArgumentParser(description="Specify the port and address of the server",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a",default=SERVER_ADDRESS,type=str,help="Server address")
parser.add_argument("-p",default=PORT_NUMBER,type=int,help="Server port")
args = parser.parse_args()

SERVER_ADDRESS = args.a
if not (SERVER_ADDRESS.startswith("http://") or SERVER_ADDRESS.startswith("https://")):
    SERVER_ADDRESS = f"http://{SERVER_ADDRESS}"
PORT_NUMBER = args.p

#If anyone says your client is using someone else's server after you run this, show this line
if os.system(f"ss -ltnup | grep ':{PORT_NUMBER}' | grep -o 'pid=[0-9]*' | grep -o '[0-9]*'") == 0:
    print("Another process is using the current port. Terminating...")
    os.system(f"kill -9 `ss -ltnup | grep ':{PORT_NUMBER}' | grep -o 'pid=[0-9]*' | grep -o '[0-9]*'`")

#Establish session variables
MAIN_MODEL = "YOLOv11 Segmentation M (Augmented) (Best)"
SUB_MODEL = "YOLOv11 Segmentation N (Augmented) (Best)"
MODELS_DIR = os.path.join(os.getcwd(), "../","models-complete")
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LIST = list(range(torch.cuda.device_count()))+["cpu"]
MAX_FRAME_RATE = 30
session_data = {}
#store the model and model dict separately as they are used by all sessions
model_dict = {}
model = {}
#Allow connections for open ports
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://localhost:8004",
    "http://localhost:8005",
    "http://localhost:8006",
    "http://localhost:8007",
    "http://localhost:8008",
    "http://localhost:8009",
    "http://localhost:8010",
    "https://localhost",
    "https://localhost:8000",
    "https://localhost:8001",
    "https://localhost:8002",
    "https://localhost:8003",
    "https://localhost:8004",
    "https://localhost:8005",
    "https://localhost:8006",
    "https://localhost:8007",
    "https://localhost:8008",
    "https://localhost:8009",
    "https://localhost:8010",
]



def load_model(model_name=MAIN_MODEL,model=model,model_dict=model_dict,model_id=0):
    #Build dict of available models on first run
    if len(model_dict) == 0:
        for model_dir in os.listdir(MODELS_DIR):
            logger.info("Model found: %s",model_dir)
            weight_path = os.path.join(MODELS_DIR, model_dir, "weights", "best.pt")
            if os.path.isfile(weight_path):
                logger.info("Best model path: %s (Best)",weight_path)
                model_dict[model_dir+" (Best)"] = weight_path
            weight_path = os.path.join(MODELS_DIR, model_dir, "weights", "last.pt")
            if os.path.isfile(weight_path):
                logger.info("Last model path: %s (Last)",weight_path)
                model_dict[model_dir+" (Last)"] = weight_path
    torch.cuda.empty_cache()
    model[model_id] = YOLO(model_dict[model_name],task="segment")
    torch.cuda.empty_cache()
    return model

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    #Fixed parameters for deployment
    #Assign default session parameters to int(0)
    session_data[0] = {
        "device":0 if torch.cuda.is_available() else "cpu",
        "conf": 0.7, #must be from 0 to 1 inclusive
        "iou": 0.2, #must be from 0 to 1 inclusive
        "show-boxes": True,
        "show-masks": True,
        "zoom-scale":1.0, #Scale must be >= 1.0. Disable zoom by setting this to 1.0
        "zoom-mode":"center", #Should be "center" or "grid",otherwise no zoom
        "rescale": False, #only has an effect if zoom is enabled
        "resize": False, #resize image before inference
        "cx":0.5, #must be within 0 to 1 exclusive, scaled to image width
        "cy":0.5, #must be within 0 to 1 exclusive, scaled to image height
    }
    load_model(MAIN_MODEL,model,model_dict,0)
    load_model(SUB_MODEL,model,model_dict,1)
    logger.info("Initialized variables!")
    logger.info("Current model: %s",model)
    logger.info("Model dict: %s",model_dict)
    logger.info("Current parameters: %s",session_data[0])
    yield
    session_data.clear()
    model_dict.clear()
    model.clear()
    torch.cuda.empty_cache()
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=["*"],allow_headers=["*"],)


# setup static and template
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

#send the list of available models to the client
@app.get("/get_models")
async def get_models():
    return JSONResponse(content=jsonable_encoder(model_dict.keys()))

#send the list of available devices to the client
@app.get("/get_devices")
async def get_devices():
    return JSONResponse(content=jsonable_encoder(DEVICE_LIST))


@app.put("/update_session_var/{session_token}/{option}/{var}",response_model=Message, responses={404: {"model": Message}})
async def update_var(session_token:str,option: str, var: str):
    logger.info("UPDATING SESSION %s VARIABLE %s to %s",session_token,option,var)
    if session_token not in session_data:
        logger.info("Session %s not present! Updating...",session_token)
        session_data[session_token] = session_data[0].copy()
    #Remap values to their corresponding variables
    logger.info("Previous value %s has value %s with type %s",option,session_data[session_token].get(option,None),type(session_data[session_token].get(option,None)))
    match option:
        case "conf":
            session_data[session_token]["conf"] = float(var)
        case "iou":
            session_data[session_token]["iou"] = float(var)
        case "zoom-scale":
            session_data[session_token]["zoom-scale"] = float(var)
        case "cx":
            session_data[session_token]["cx"] = float(var)
        case "cy":
            session_data[session_token]["cy"] = float(var)
        case "show-boxes":
            session_data[session_token]["show-boxes"] = (var.lower().split("=",1)[1] == "true")
        case "show-masks":
            session_data[session_token]["show-masks"] = (var.lower().split("=",1)[1] == "true")
        case "resize":
            session_data[session_token]["resize"] = (var.lower().split("=",1)[1] == "true")
        case "rescale":
            session_data[session_token]["rescale"] = (var.lower().split("=",1)[1] == "true")
        case "zoom-mode":
            session_data[session_token]["zoom-mode"] = var.lower().split("=",1)[0]
        case _:
            return JSONResponse(status_code=404, content={"message": f"Parameter {option} must be in {session_data[session_token].keys()}"})
    logger.info("Current value %s has value %s with type %s",option,session_data[session_token][option],type(session_data[session_token][option]))
    return JSONResponse(status_code=200, content={"message": "Session variables updated!"})







# Use YOLO model to infer image
def infer_image(image,session_token=0,persist=True):
    def get_box(w,h,session_token=0):
        w_s,h_s = w/session_data[session_token]["zoom-scale"],h/session_data[session_token]["zoom-scale"]
        cw,ch = w*session_data[session_token]["cx"],h*session_data[session_token]["cy"]
        left_bound,right_bound = int(cw-(w_s/2)),int(cw+(w_s/2))
        top_bound,bottom_bound = int(ch-(h_s/2)),int(ch+(h_s/2))
        #Ensure that the zoom box does not exceed the boundaries of the image stream
        #Since the zoom box is smaller than the image, adjust cx and cy to fit the scale
        if left_bound < 0:
            right_bound -= left_bound
            cw -= left_bound
            left_bound = 0
        if right_bound > w:
            left_bound -= (right_bound-w)
            cw -= (right_bound-w)
            right_bound = w
        if top_bound < 0:
            bottom_bound -= top_bound
            ch -= top_bound
            top_bound = 0
        if bottom_bound > h:
            top_bound -= (bottom_bound-h)
            ch -= (bottom_bound-h)
            bottom_bound = h
        return int(left_bound),int(top_bound),int(right_bound),int(bottom_bound),int(cw),int(ch),int(w_s),int(h_s)

    def zoom_image(image,session_token=0):
        w,h = image.shape[1],image.shape[0]
        left_bound,top_bound,right_bound,bottom_bound,cw,ch,w_s,h_s = get_box(w,h,session_token)
        logger.info("ZOOM DIMS: %d|%d || CENTER: %d|%d",w,h,cw,ch)
        logger.info("BOUNDS: (%d,%d),(%d,%d)",left_bound,top_bound,right_bound,bottom_bound)
        new_img = image[max(0,top_bound):min(bottom_bound,h),max(0,left_bound):min(right_bound,w)]
        new_img = cv2.resize(new_img,dsize=(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
        return new_img

    def unzoom_image(image,session_token=0):
        if not isinstance(image,np.ndarray):
            image = image.numpy()
        mask_img = np.zeros_like(image)
        
        w,h = image.shape[1],image.shape[0]
        left_bound,top_bound,right_bound,bottom_bound,_,_,w_s,h_s = get_box(w,h,session_token)
        new_img = cv2.resize(image,dsize=(int(w_s),int(h_s)),interpolation=cv2.INTER_NEAREST_EXACT)
        mask_img[int(max(0,top_bound)):int(min(bottom_bound,h)),int(max(0,left_bound)):int(min(right_bound,w))] = new_img
        return torch.from_numpy(mask_img)

    if session_data[session_token]["resize"]:
        if image.shape[1] >= image.shape[0]:
            reduced_image = cv2.resize(image,(640,int(image.shape[0]*640/image.shape[1])),interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            reduced_image = cv2.resize(image,(int(image.shape[1]*640/image.shape[0]),640),interpolation=cv2.INTER_LINEAR_EXACT)
        results = model[0].track(reduced_image,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)
        results[0].orig_img = image
        results[0].orig_shape = image.shape[:2]

        #Resize annotations for bounding boxes and masks
        def rescale1_x(x):
            return int((x*image.shape[1]/reduced_image.shape[1]))
        def rescale1_y(y):
            return int((y*image.shape[0]/reduced_image.shape[0]))
        boxes = None if results[0].boxes is None else results[0].boxes.cpu().data.clone()
        if boxes is not None:
            for row in boxes:
                row[0] = rescale1_x(row[0])
                row[1] = rescale1_y(row[1])
                row[2] = rescale1_x(row[2])
                row[3] = rescale1_y(row[3])
        masks = None if results[0].masks is None else results[0].masks.cpu().data.clone()
        if masks is not None:
            new_masks = torch.zeros((masks.shape[0],image.shape[0],image.shape[1]))
            for i,row in enumerate(masks):
                new_mask = cv2.resize(row.numpy(),dsize=(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST_EXACT)
                new_masks[i] = torch.from_numpy(new_mask) #rescale masks
            masks = new_masks
        results[0].update(boxes=boxes,masks=masks) #update rescaled boxes and masks
    else:
        results = model[0].track(image,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)
    #Use 2x zoom if no objects were found
    if not results[0].boxes:
        logger.info("No objects found.")
        if session_data[session_token]["zoom-scale"] > 1.0 and session_data[session_token]["zoom-mode"] != "disable":
            logger.info("Zooming in...")

            #Remove this line in production! - CONFIRM ZOOM LOCATION
            #w,h = image.shape[1],image.shape[0]
            #left_bound,top_bound,right_bound,bottom_bound,cw,ch,_,_ = get_box(w,h,session_token)
            #image = cv2.rectangle(image,(int(max(0,left_bound)),int(max(0,top_bound))),(int(min(right_bound,w)),int(min(bottom_bound,h))),(0,255,0))
            #image = cv2.circle(image,(int(cw),int(ch)),5,(0,255,0),2)
            #end remove line

            image2 = zoom_image(image,session_token)
            #Attempt to use another lighter model if it exists
            if session_data[session_token]["resize"]:
                if image2.shape[1] >= image2.shape[0]:
                    reduced_image2 = cv2.resize(image2,(640,int(image.shape[0]*640/image.shape[1])),interpolation=cv2.INTER_LINEAR_EXACT)
                else:
                    reduced_image2 = cv2.resize(image2,(int(image.shape[1]*640/image.shape[0]),640),interpolation=cv2.INTER_LINEAR_EXACT)
                results2 = model.get(1,model[0]).track(reduced_image2,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)
                results2[0].orig_img = image2
                results2[0].orig_shape = image2.shape[:2]

                #Resize annotations for bounding boxes and masks
                def rescale2_x(x):
                    return int((x*image2.shape[1]/reduced_image2.shape[1]))
                def rescale2_y(y):
                    return int((y*image2.shape[0]/reduced_image2.shape[0]))
                boxes = None if results2[0].boxes is None else results2[0].boxes.cpu().data.clone()
                if boxes is not None:
                    for row in boxes:
                        row[0] = rescale2_x(row[0])
                        row[1] = rescale2_y(row[1])
                        row[2] = rescale2_x(row[2])
                        row[3] = rescale2_y(row[3])
                masks = None if results2[0].masks is None else results2[0].masks.cpu().data.clone()
                if masks is not None:
                    new_masks = torch.zeros((masks.shape[0],image.shape[0],image.shape[1]))
                    for i,row in enumerate(masks):
                        new_mask = cv2.resize(row.numpy(),dsize=(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST_EXACT)
                        new_masks[i] = torch.from_numpy(new_mask) #rescale masks
                    masks = new_masks
                results2[0].update(boxes=boxes,masks=masks) #update rescaled boxes and masks
            else:
                results2 = model.get(1,model[0]).track(image2,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)

            if session_data[session_token]["rescale"]:
                results2[0].orig_img = results[0].orig_img
                w,h = image.shape[1],image.shape[0]
                left_bound,top_bound,right_bound,bottom_bound,cw,ch,_,_ = get_box(w,h,session_token)
                #Remove this line in production! - CONFIRM ZOOM POSITION
                #results2[0].orig_img = cv2.rectangle(results2[0].orig_img,(int(max(0,left_bound)),int(max(0,top_bound))),(int(min(right_bound,w)),int(min(bottom_bound,h))),(0,255,0))
                #results2[0].orig_img = cv2.circle(results2[0].orig_img,(int(cw),int(ch)),5,(0,255,0),2)
                #end remove line
                #Resize annotations for bounding boxes and masks
                def rescale_x(x):
                    return int((x/session_data[session_token]["zoom-scale"])+left_bound)
                def rescale_y(y):
                    return int((y/session_data[session_token]["zoom-scale"])+top_bound)
                boxes = None if results2[0].boxes is None else results2[0].boxes.cpu().data.clone()
                if boxes is not None:
                    for row in boxes:
                        row[0] = rescale_x(row[0])
                        row[1] = rescale_y(row[1])
                        row[2] = rescale_x(row[2])
                        row[3] = rescale_y(row[3])
                masks = None if results2[0].masks is None else results2[0].masks.cpu().data.clone()
                if masks is not None:
                    for row in masks:
                        row[:] = unzoom_image(row) #rescale masks
                results2[0].update(boxes=boxes,masks=masks) #update rescaled boxes and masks
                logger.info("Showing scaled image...")
            else:
                logger.info("Showing zoomed in image...")
            final_img = results2[0].plot(boxes=session_data[session_token]["show-boxes"],masks=session_data[session_token]["show-masks"])
        else:
            final_img = results[0].plot(boxes=session_data[session_token]["show-boxes"],masks=session_data[session_token]["show-masks"])
    else:
        logger.info('Detected items!')
        final_img = results[0].plot(boxes=session_data[session_token]["show-boxes"],masks=session_data[session_token]["show-masks"])
    return final_img



#Obtain the processed image via websocket
@app.websocket('/ws_image_processing/{session_token}')
async def ws_image_processing(websocket: WebSocket, session_token: str):
    #Define image getter/setter threads
    async def get_image(websocket: WebSocket, image_queue: asyncio.Queue):
        while True:
            image_bytes = await websocket.receive_bytes()
            try:
                #Record the time that the image was received
                image_queue.put_nowait((image_bytes,time.perf_counter()))
            except asyncio.QueueFull:
                pass
    async def process_image(websocket: WebSocket, image_queue: asyncio.Queue,session_token=0):
        while True:
            #Start timing on receiving an image
            image_bytes,ti = await image_queue.get()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            #Debug: verify parameters
            logger.info("Session token is %s with type %s",session_token,type(session_token))
            logger.info("Current parameters: %s",session_data[session_token])
            #Perform model inference
            final_img = infer_image(image,session_token)
            #Return the image to the client
            _, byte_data = cv2.imencode(".jpg", final_img)
            final_img_str = base64.encodebytes(byte_data).decode('utf-8')
            processed_img_data = "data:image/jpg;base64," + final_img_str
            await websocket.send_text(processed_img_data)
            #Finish timing on sending the image to the client
            tf = time.perf_counter()
            logger.info("Inference time: %f ms",tf-ti)
            logger.info("FPS: %f",tf-ti and 1/(tf-ti) or 0.0)

    #Start the connection
    await websocket.accept()
    logger.info("Connected to session with token: %s",session_token)
    #initialize session parameters on connection
    if session_token not in session_data:
        session_data[session_token] = session_data[0].copy()
    logger.info("Initialized parameters for session %s",session_token)
    #Save only the most recent FRAME_RATE/3 images for processing
    image_queue = asyncio.Queue(maxsize=max(1,round(MAX_FRAME_RATE/3)))
    logger.info("Started process stream for session %s", session_token)
    #Parse and infer the webcam stream
    get_images_task = asyncio.create_task(get_image(websocket, image_queue))
    process_images_task = asyncio.create_task(process_image(websocket, image_queue, session_token))
    try:
        done, pending = await asyncio.wait(
            {get_images_task, process_images_task},
            return_when=asyncio.FIRST_COMPLETED
        )
        logger.info("Image processing for session %s complete!", session_token)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
        
    except (WebSocketDisconnect, ConnectionClosed):
        # Remove session data when user disconnects
        logger.info('Session %s has disconnected.',session_token)
        del session_data[session_token]
        logger.info("Removed session data for session %s", session_token)

#Obtain the processed image via requests
@app.post("/image_processing")
async def image_processing(file: UploadFile = File(...)):
    image_bytes = await file.read() #read the sent image
    #Perform inference on the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result_frame = await infer_image(image)
    _, buffer = cv2.imencode('.jpg', result_frame)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_NUMBER,ssl_keyfile='credentials/key.pem',ssl_certfile='credentials/cert.pem')
