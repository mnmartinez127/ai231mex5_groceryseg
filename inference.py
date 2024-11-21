import os
import time
import cv2
import requests
import av
import streamlit as st
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from ultralytics import YOLO
import numpy as np
import torch

SERVER_ADDRESS = "http://202.92.159.241:8086/infer_frame"
#SERVER_ADDRESS = "http://127.0.0.1:8086/infer_frame"
RTC_CONFIG =  [{"urls": ["stun:stun.l.google.com:19302"]}]
MAX_FRAME_RATE = 60

#Model data
root_dir = os.path.join(os.getcwd())#,"..")
models_dir = os.path.join(root_dir, "models")
model_dict = {}
#get the last model from training
for model_dir in os.listdir(models_dir):
    weight_path = os.path.join(models_dir, model_dir, "weights", "last.pt")
    if os.path.exists(weight_path):
        model_dict[model_dir] = weight_path


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
show_boxes = st.sidebar.checkbox("Show Boxes", value = True)
show_masks = st.sidebar.checkbox("Show Masks", value = True)

source_select = st.sidebar.radio(
    "Input Source", ["Webcam", "Webcamv2", "Image"]
)
mode_select = st.sidebar.radio("Infer Source", ["Server","Client"])

#Select the model to be used
model_select = st.sidebar.radio("Select Model", sorted(list(model_dict.keys())))



class Model:
    model:YOLO|None
    params:dict
    categories:list
    def __init__(self,params:dict={}):
        self.model = None
        self.params = {}
        self.categories = []
        self.update_params(params)

    def update_params(self,params):
        if params.get("model_name","") != self.params.get("model_name",""):
            torch.cuda.empty_cache()
            if params.get("model_name","").lower().startswith("yolo"):
                try:
                    self.model = YOLO(model_dict.get(params.get("model_name",""),"yolo11n-seg.pt"),task="segment")
                    self.categories = [] if self.model.names is None else sorted(self.model.names.items(), key = lambda x:x[0])
                except Exception:
                    pass
        self.params.update(params)

    def infer(self,image,params:dict={},address:str=""):
        if address: #server-side processing
            print("Performing server-side inference!")
            err, encoded_image = cv2.imencode('.jpg', image)
            sent_params = {}
            sent_params.update(self.params)
            sent_params.update(params)
            response = requests.post(address,params=sent_params,files={'file': encoded_image.tobytes()},timeout=1.0)
            result_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            print("Performed server-side inference!")
        else: #client side processing
            print("Performing client-side inference!")
            self.update_params(params)
            if self.model == None:
                return image
            results = self.model.track(image,
                                        conf = self.params.get("conf_threshold",0.0),
                                        iou = self.params.get("iou_threshold",0.0),
                                        classes = self.params.get("classes",None)
                                        )
            result_frame =  results[0].plot(boxes = self.params.get("show_boxes",True),
                                                masks = self.params.get("show_masks",True)
                                                )
            print("Performed client-side inference!")
        return result_frame


params = {
    "model_name": model_select,
}
model = Model(params)

#Use a 2-row display instead of a sidebar
display_columns = st.columns(2)
with display_columns[0]:
    st.header("Options")
    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.multiselect("Categories", model.categories, format_func = lambda x: f"{x[0]}: {x[1]}", placeholder = "All classes")
    conf_threshold = float(st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01))
    iou_threshold = float(st.slider("IoU Threshold", 0.0, 1.0, 0.1, 0.01))


new_params = {
    "model_name": model_select,
    "conf": conf_threshold,
    "iou": iou_threshold,
    "classes": [i[0] for i in selected_classes] if selected_classes else None,
    "show_boxes": show_boxes,
    "show_masks": show_masks,
}

address = SERVER_ADDRESS if mode_select == "Server" else ""
with display_columns[1]:
    match source_select:

        case "Webcam":
            class InferenceProcessor(VideoProcessorBase):
                model: Model
                params: dict
                address: str
                frame_no: int
                frame_queue: "queue.Queue[int]"
                def __init__(self):
                    self.model = Model()
                    self.params = {}
                    self.address = ""
                    self.frame_no = 0
                    self.frame_queue = queue.Queue()
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    try:
                        image = frame.to_ndarray(format = "bgr24")
                        result_frame = self.model.infer(image,self.params,self.address)
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
                        infer_stream.video_processor.model = model
                        infer_stream.video_processor.params = new_params
                        infer_stream.video_processor.address = address
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
                        image = frame.to_ndarray(format = "bgr24")
                        result_frame = model.infer(image,new_params,address)
                        image_place.image(cv2.cvtColor(result_frame,cv2.COLOR_RGB2BGR))
                        tf = time.time()
                        fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
                        fps_text.write(f"FPS: {fps:.2f}")
                        ti = tf
                else:
                    print("Ending inference.")
                    break

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
                result_frame = model.infer(image,new_params,address)
                st.text("Results")
                st.image(result_frame, caption = "Result Image", use_column_width = True, channels = "BGR", output_format = "auto")
            else:
                st.text("No Uploaded Image")
                st.text("No Results")


        case _:
            st.error("Error: No input source selected!")
