import os
import argparse
import time
import queue
import requests
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    import av
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
except ImportError:
    os.system("pip install opencv-python opencv-python-contrib")
    os.system("pip install 'numpy<2.0'")
    os.system("pip install ultralytics")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install streamlit streamlit-webrtc")
    #Best to install these manually
    #os.system("pip install onnx onnxslim onnxruntime-gpu")
    #os.system("pip install tensorrt")


#Define initial constants, should not change!
SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
PORT_NUMBER = 8002 #inference server
#PORT_NUMBER = 8003 #test server
parser = argparse.ArgumentParser(description="Specify the port and address of the server",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a",default=SERVER_ADDRESS,help="Server address")
parser.add_argument("-p",default=PORT_NUMBER,help="Server port")
args = parser.parse_args()
SERVER_ADDRESS = args.a
if not (SERVER_ADDRESS.startswith("http://") or SERVER_ADDRESS.startswith("https://")):
    SERVER_ADDRESS = f"http://{SERVER_ADDRESS}"
PORT_NUMBER = args.p
LIST_PAGE = "/get_lists"
INFER_PAGE = "/infer_frame"
SERVER_URL = f"{SERVER_ADDRESS}:{PORT_NUMBER}{INFER_PAGE}"
LIST_SERVER_URL = f"{SERVER_ADDRESS}:{PORT_NUMBER}{LIST_PAGE}"

RTC_CONFIG =  [{"urls": ["stun:stun.l.google.com:19302"]}]
MAX_FRAME_RATE = 60
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
    "model_name":"No Model",
    "model_device":0 if torch.cuda.is_available() else "cpu",
    "model_mode":"torch",
    "server_url":SERVER_URL,
    "conf":0.7,
    "iou":0.1,
    "classes":[],
    "show_boxes":True,
    "show_masks":True,
}

server_lists = {
    "models":MODEL_LIST.copy(),
    "devices":DEVICE_LIST.copy(),
    "modes":MODE_LIST.copy()
}

class Model:
    model:YOLO|None
    params:dict
    def __init__(self,params):
        self.model = None
        self.params = {
        "model_name":"No Model",
        "model_device":0 if torch.cuda.is_available() else "cpu",
        "model_mode":"torch",
        "server_url":SERVER_URL,
        "conf":0.5,
        "iou":0.1,
        "classes":[],
        "show_boxes":True,
        "show_masks":True,
        }
        self.update_params(params)

    def update_params(self,params):
        #Load model if it changed
        if self.params["model_name"] != params["model_name"] or self.params["model_mode"] != params["model_mode"]:
            try:
                torch.cuda.empty_cache()
                model_path = model_dict.get(params["model_name"],"")
                if not os.path.isfile(model_path):
                    model = None
                    classes = []
                else:
                    match params["model_mode"]:
                        case "onnx":
                            if os.path.exists(os.path.splitext(model_path)[0]+".onnx"):
                                model = YOLO(os.path.splitext(model_path)[0]+".onnx",task="segment")
                            else:
                                model = YOLO(model_path,task="segment")
                                new_model_path = model.export(format="onnx",device=params["model_device"])
                                model = YOLO(new_model_path,task="segment")
                        #case "tensorrt":
                        #    if os.path.exists(os.path.splitext(model_path)[0]+".engine"):
                        #        model = YOLO(os.path.splitext(model_path)[0]+".engine",task="segment")
                        #    else:
                        #        model = YOLO(model_path,task="segment")
                        #        new_model_path = model.export(format="engine",device=params["model_device"])
                        #        model = YOLO(new_model_path,task="segment")
                        case _:
                                model = YOLO(model_path,task="segment")
                    classes = [] if model.names is None else sorted(model.names.items(), key = lambda x:x[0])
            except Exception:
                pass
            else:
                self.model = model
                self.params["classes"] = classes
            finally:
                torch.cuda.empty_cache()
        #update parameters
        self.params.update(params)

    def infer_image(self,image,params):
        if params.get("server_url",self.params["server_url"]): #server-side processing
            sent_params = {}
            sent_params.update(self.params)
            sent_params.update(params)
            address = sent_params.pop("server_url") #do not include address in request
            print(f"Performing server-side inference! Server is {address}")
            #Encode image before sending to server
            err, encoded_image = cv2.imencode('.jpg', image)
            response = requests.post(address,params=sent_params,files={'file': encoded_image.tobytes()})
            result_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            print("Performed server-side inference!")
        else: #client side processing
            print("Performing client-side inference!")
            self.update_params(params)
            if self.model == None: #do nothing if model is not loaded
                return image
            results = self.model.track(image,
                                        conf = self.params["conf"],
                                        iou = self.params["iou"],
                                        classes = self.params["classes"],
                                        device = self.params["model_device"],
                                        )
            result_frame =  results[0].plot(boxes = self.params["show_boxes"],
                                                masks = self.params["show_masks"]
                                                )
            print("Performed client-side inference!")
        return result_frame

#Get model, mode, and device lists from server
def get_server_lists(server_lists):
    response = requests.get(LIST_SERVER_URL).json()
    server_lists.update(response)
    print(server_lists)
    return server_lists

#initialize model
model = Model(params)


#CLIENT: Streamlit frontend

st.set_page_config(
    page_title = "Grocery Instance Segmentation", 
    page_icon = "🛒", 
    layout = "centered", 
    initial_sidebar_state = "collapsed"
)
st.title("Machine Exercise 5")

#advanced options
st.sidebar.header("Advanced Options")
#Select the input source
source_select = st.sidebar.radio("Input Source", ["Webcam", "Webcamv2", "Image"])

server_select = st.sidebar.radio("Infer Source", ["Server","Client"])

server_lists = get_server_lists(server_lists)
if server_select == "Server":
    device_select = st.sidebar.radio("Device Used", server_lists["devices"]+["cpu"])
    mode_select = st.sidebar.radio("Model Mode",server_lists["modes"])
    model_select = st.sidebar.radio("Select Model", server_lists["models"]+["No Model"])
else: #"Client" option
    device_select = st.sidebar.radio("Device Used", DEVICE_LIST+["cpu"])
    mode_select = st.sidebar.radio("Model Mode",MODE_LIST)
    model_select = st.sidebar.radio("Select Model", MODEL_LIST+["No Model"])


#Use a 2-row display instead of a sidebar
display_columns = st.columns(2)
with display_columns[0]:
    st.header("Options")
    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.multiselect("classes", model.params["classes"], format_func = lambda x: f"{x[0]}: {x[1]}", placeholder = "All classes")
    conf_threshold = float(st.slider("Confidence Threshold", 0.0, 1.0, round(params["conf"],2), 0.01))
    iou_threshold = float(st.slider("IoU Threshold", 0.0, 1.0, round(params["iou"],2), 0.01))
    show_boxes = st.checkbox("Show Boxes", value = True)
    show_masks = st.checkbox("Show Masks", value = True)

#get UI parameters
new_params = {
    "model_name": model_select,
    "model_device": device_select,
    "model_mode": mode_select,
    "server_url":SERVER_URL if server_select == "Server" else "",
    "conf": conf_threshold,
    "iou": iou_threshold,
    "classes": [i[0] for i in selected_classes] if selected_classes else None,
    "show_boxes": show_boxes,
    "show_masks": show_masks,
}
#Update session parameters
params.update(new_params)

#Inference Window
with display_columns[1]:
    match source_select:
        case "Webcam":
            class InferenceProcessor(VideoProcessorBase):
                model: Model
                params: dict
                frame_no: int
                frame_queue: "queue.Queue[int]"
                def __init__(self):
                    self.params = {
                    "model_name":"No Model",
                    "model_device":0 if torch.cuda.is_available() else "cpu",
                    "model_mode":"torch",
                    "server_url":SERVER_URL,
                    "conf":0.5,
                    "iou":0.1,
                    "classes":[],
                    "show_boxes":True,
                    "show_masks":True,
                    }
                    self.model = Model(self.params)
                    self.frame_no = 0
                    self.frame_queue = queue.Queue()
                #Inference function upon receiving each frame
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        image = frame.to_ndarray(format = "bgr24")
                        result_frame = self.model.infer_image(image,self.params)
                        output_frame = av.VideoFrame.from_ndarray(result_frame, format = "bgr24")
                        self.frame_queue.put(self.frame_no)
                        self.frame_no += 1
                        return output_frame
                    except Exception as e:
                        print(e)
                        return frame

            infer_stream = webrtc_streamer(
                key = "Inference Result", 
                mode = WebRtcMode.SENDRECV, 
                rtc_configuration = RTC_CONFIG, 
                video_processor_factory = InferenceProcessor, 
                media_stream_constraints = {"video": True, "audio": False}, 
                async_processing = True, 
            )
            fps_text = st.text("FPS: None")
            fps = 0.0
            ti = time.time()
            if infer_stream.state.playing:
                while True:
                    if infer_stream.video_processor:
                        infer_stream.video_processor.params.update(new_params)
                        #Calculate FPS, inference loop is inside infer_stream
                        try:
                            frame_no = infer_stream.video_processor.frame_queue.get(timeout = 1.0)
                        except queue.Empty:
                            print("No image found!")
                            fps_text.write("FPS: 0.00")
                        else:
                            tf = time.time()
                            fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
                            print("Image found!")
                            fps_text.write(f"FPS: {fps:.2f}")
                            ti = tf
                    else:
                        print("Ending inference.")
                        break
        #Alternate implementation that does inference outside the infer_stream thread
        case "Webcamv2":
            infer_stream = webrtc_streamer(
                key = "Inference Result", 
                mode = WebRtcMode.SENDONLY, 
                rtc_configuration = RTC_CONFIG, 
                media_stream_constraints = {"video": True, "audio":False}, 
            )

            image_place = st.empty()
            fps_text = st.text("FPS: None")
            fps = 0.0
            ti = time.time()
            while True:
                if infer_stream.video_receiver:
                    try:
                        frame = infer_stream.video_receiver.get_frame(timeout = 1.0)
                    except queue.Empty:
                        print("No image found.")
                        fps_text.write("FPS: 0.00")
                    else:
                        print("Image found.")
                        #perform inference
                        image = frame.to_ndarray(format = "bgr24")
                        result_frame = model.infer_image(image,new_params)
                        image_place.image(cv2.cvtColor(result_frame,cv2.COLOR_RGB2BGR))
                        #calculate FPS
                        tf = time.time()
                        fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
                        fps_text.write(f"FPS: {fps:.2f}")
                        ti = tf
                else:
                    print("Ending inference.")
                    break
        #Upload a single image
        case "Image":
            img = st.file_uploader("Upload Image", type = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"])
            if img is not None:
                #cache image to disk
                cache_dir = os.path.join(os.getcwd(), "___cache")
                cache_path = os.path.join(cache_dir, img.name)
                os.makedirs(cache_dir, exist_ok = True)
                with open(cache_path, "wb") as f:
                    f.write(img.getbuffer())
                image = cv2.imread(cache_path)
                st.text("Uploaded image")
                st.image(image, caption = "Input Image", use_column_width = True, channels = "BGR", output_format = "auto")
                result_frame = model.infer_image(image,new_params)
                st.text("Results")
                st.image(result_frame, caption = "Result Image", use_column_width = True, channels = "BGR", output_format = "auto")
            else:
                st.text("No Uploaded Image")
                st.text("No Results")


        case _:
            st.error("Error: No input source selected!")
