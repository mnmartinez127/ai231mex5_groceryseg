import logging
import asyncio
import contextlib
import cv2
import numpy as np
from torch import cuda as tc, from_numpy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Boxes,Masks
import base64
import threading
import time
from websockets.exceptions import ConnectionClosed
from fastapi import (FastAPI, Request, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, Header)
from fastapi.responses import HTMLResponse, JSONResponse
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
session_data = {}
model = {}
params = {}
DEVICE_LIST = list(range(tc.device_count()))+["cpu"]

# dependencies injection
def get_session_token(x_session_token: str = Header(...)):
    return x_session_token

def get_current_session_vars(token: str = Depends(get_session_token)):
    # Check if session exists, otherwise create a new one
    if token not in session_data:
        session_data[token] = {"bg_img": None, "cartoonify": None}
    return session_data[token]

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    #Fixed parameters for deployment
    model[0] = YOLO("best.pt",task="segment")
    params["device"] = 0 if tc.is_available() else "cpu"

    #UI parameters for deployment
    params["conf"] = 0.6
    params["iou"] = 0.1
    params["show_boxes"] = True
    params["show_masks"] = True
    params["scale"] = 2.0
    params["cx"] = 0.5
    params["cy"] = 0.5
    yield

app = FastAPI(lifespan=lifespan)

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
]

app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=["*"],allow_headers=["*"],)

# setup static and template
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})






# Use YOLO model to infer image
def infer_image(image):
    def zoom_image(image):
        w,h = image.shape[1],image.shape[0]
        ws,hs = w*params["scale"],h*params["scale"]
        cw,ch = ws*params["cx"],hs*params["cy"]
        left_bound,right_bound = int(cw-(w/2)),int(cw+(w/2))
        top_bound,bottom_bound = int(ch-(h/2)),int(ch+(h/2))
        logger.info(f"DIMS: {w}|{h} ||| CENTER: {cw}|{ch}")
        logger.info(f"BOUNDS: {left_bound}|{right_bound}|{top_bound}|{bottom_bound}")


        new_img = cv2.resize(image,dsize=(int(ws),int(hs)),interpolation=cv2.INTER_NEAREST_EXACT)
        #Remove this line in production!
        #new_img = cv2.rectangle(new_img,(int(max(0,left_bound)),int(max(0,top_bound))),(int(min(right_bound,ws)),int(min(bottom_bound,hs))),(255,0,0))
        #new_img = cv2.circle(new_img,(int(cw),int(ch)),5,(255,0,0),2)
        #end remove line
        new_img = new_img[max(0,top_bound):min(bottom_bound,hs),max(0,left_bound):min(right_bound,ws)]
        return new_img

    def unzoom_image(image):
        if not isinstance(image,np.ndarray):
            image = image.numpy()
        mask_img = np.zeros_like(image)
        w,h = image.shape[1],image.shape[0]
        w_s,h_s = w/params["scale"],h/params["scale"]
        cw,ch = w*params["cx"],h*params["cy"]
        left_bound,right_bound = int(cw-((w/params["scale"])/2)),int(cw+((w/params["scale"])/2))
        top_bound,bottom_bound = int(ch-((h/params["scale"])/2)),int(ch+((h/params["scale"])/2))
        new_img = cv2.resize(image,dsize=(int(w_s),int(h_s)),interpolation=cv2.INTER_NEAREST_EXACT)
        mask_img[int(max(0,top_bound)):int(min(bottom_bound,h)),int(max(0,left_bound)):int(min(right_bound,w))] = new_img
        return from_numpy(mask_img)

    results = model[0].track(image,conf=params["conf"],iou=params["iou"],device=params["device"])
    #Use 2x zoom if no objects were found
    if not results[0].boxes:
        image2 = zoom_image(image)
        results2 = model[0].track(image2,conf=params["conf"],iou=params["iou"],device=params["device"])

        if True:
            results2[0].orig_img = results[0].orig_img

            w,h = image.shape[1],image.shape[0]
            w_s,h_s = w/params["scale"],h/params["scale"]
            cw,ch = w*params["cx"],h*params["cy"]
            left_bound,right_bound = int(cw-((w/params["scale"])/2)),int(cw+((w/params["scale"])/2))
            top_bound,bottom_bound = int(ch-((h/params["scale"])/2)),int(ch+((h/params["scale"])/2))
            #Remove this line in production!
            results2[0].orig_img = cv2.rectangle(results2[0].orig_img,(int(max(0,left_bound)),int(max(0,top_bound))),(int(min(right_bound,w)),int(min(bottom_bound,h))),(0,255,0))
            results2[0].orig_img = cv2.circle(results2[0].orig_img,(int(cw),int(ch)),5,(0,255,0),2)
            #end remove line
            #Resize annotations for bounding boxes and masks
            rescale_x = lambda x: (int((x/params["scale"])+left_bound))
            rescale_y = lambda y: (int((y/params["scale"])+top_bound))
            boxes = None if results2[0].boxes is None else results2[0].boxes.cpu().data.clone()
            logger.info(f"boxes:{boxes}")
            if boxes is not None:
                for row in boxes:
                    row[0] = rescale_x(row[0])
                    row[1] = rescale_y(row[1])
                    row[2] = rescale_x(row[2])
                    row[3] = rescale_y(row[3])

            masks = None if results2[0].masks is None else results2[0].masks.cpu().data.clone()
            logger.info(f"masks:{masks}")
            if masks is not None:
                for row in masks:
                    row[:] = unzoom_image(row)
            results2[0].update(boxes=boxes,masks=masks)

            boxes = [] if results2[0].boxes is None else results2[0].boxes.data
            masks = [] if results2[0].masks is None else results2[0].masks.data
            logger.info(f"new boxes:{boxes}")
            logger.info(f"new masks:{masks}")


        final_img = results2[0].plot(boxes=params["show_boxes"],masks=params["show_masks"])
        
    else:
        logger.info(f'Detected items!')
        final_img = results[0].plot(boxes=params["show_boxes"],masks=params["show_masks"])
    return final_img

@app.websocket('/image_processing/{session_token}')
async def ws_image_processing(websocket: WebSocket, session_token: str):
    await websocket.accept()
    logger.info(f'session token: {session_token}')
    session_vars = get_current_session_vars(session_token)
    while True:
        try:
            ti = time.perf_counter()
            bytes = await websocket.receive_bytes()
            image_array = np.frombuffer(bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            #ADD MODEL INFERENCE HERE!
            final_img = infer_image(image)
            #END MODEL INFERENCE
            _, byte_data = cv2.imencode(".jpg", final_img)
            final_img_str = base64.encodebytes(byte_data).decode('utf-8')
            processed_img_data = "data:image/jpg;base64," + final_img_str
            await websocket.send_text(processed_img_data)
            tf = time.perf_counter()
            logger.info(f"Inference time: {tf-ti and 1/(tf-ti) :.0f}ms")
        except (WebSocketDisconnect, ConnectionClosed):
            logger.info('User disconnected')
            # Remove session data when user disconnects
            del session_data[session_token]
            logger.info(f'Removed session data of session token: {session_token}')
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003,ssl_keyfile='credentials/key.pem',ssl_certfile='credentials/cert.pem')
