import os
import time
import cv2
import av
import streamlit as st
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
from ultralytics import YOLO, SAM
import numpy as np
import torch
MAX_FRAME_RATE = 60
COMMON_RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_ONNX = False
#Functions used in the program
#Load and export model
def load_model(model_select):
    #Load and export model
    torch.cuda.empty_cache()
    if os.path.exists(models[model_select]+".onnx") and USE_ONNX:
        if model_select.lower().startswith("yolo"):
            model = YOLO(models[model_select]+".onnx",task="segment")
        elif model_select.lower().startswith("sam"):
            model = SAM(models[model_select]+".onnx")
        else:
            model = None
    elif os.path.exists(models[model_select]+".pt"):
        #Convert model to ONNX format first
        if model_select.lower().startswith("yolo"):
            model = YOLO(models[model_select]+".pt",task="segment")
        elif model_select.lower().startswith("sam"):
            model = SAM(models[model_select]+".pt")
        else:
            model = None
        if USE_ONNX and model:
            try:
                model_path = model.export(format="onnx")
            except Exception:
                model_path = models[model_select]+".pt"
            else:
                models[model_select] = model_path
            finally:
                torch.cuda.empty_cache()
            if model_select.lower().startswith("yolo"):
                model = YOLO(model_path,task="segment")
            elif model_select.lower().startswith("sam"):
                model = SAM(model_path)
            else:
                model = None
    else:
        model = None
    return model

def plot_results(results, show_image = True, show_boxes = True, show_masks = True):
    if not show_image: #plot on a blank image instead of on the input
        results[0].orig_img = np.zeros_like(results[0].orig_img.shape)
    return results[0].plot(boxes = show_boxes, masks = show_masks)

class InferenceProcessor(VideoProcessorBase):
    model: YOLO|SAM|None
    conf_threshold: float
    iou_threshold: float
    classes_list: list
    show_image: bool
    show_boxes: bool
    show_masks: bool
    frame_no: int
    frame_queue: "queue.Queue[int]"
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        self.classes_list = []
        self.show_image = True
        self.show_boxes = True
        self.show_masks = True
        self.frame_no = 0
        self.frame_queue = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format = "bgr24")
            results = model.track(image, conf = self.conf_threshold, iou = self.iou_threshold, classes = self.classes_list)
            result_frame = plot_results(results, self.show_image, self.show_boxes, self.show_masks)
            output_frame = av.VideoFrame.from_ndarray(result_frame, format = "bgr24")
            self.frame_queue.put(self.frame_no)
            self.frame_no += 1
            return output_frame
        except Exception as e:
            print(e)
            return frame








#Model data
models_dir = os.path.join(os.getcwd(), "..", "models")
models = {}
#get the last model from training
for model_dir in os.listdir(models_dir):
    weight_path = os.path.join(models_dir, model_dir, "weights", "epoch499")
    if os.path.exists(weight_path+".pt") or (os.path.exists(weight_path+".onnx") and USE_ONNX):
        models[model_dir] = weight_path

#Streamlit frontend

st.set_page_config(
    page_title = "Grocery Instance Segmentation", 
    page_icon = "🛒", 
    layout = "centered", 
    initial_sidebar_state = "collapsed"
)
st.title("Machine Exercise 5")

#advanced options
st.sidebar.header("Advanced Options")

show_image = st.sidebar.checkbox("Show Image", value = True)
show_boxes = st.sidebar.checkbox("Show Boxes", value = True)
show_masks = st.sidebar.checkbox("Show Masks", value = True)

source_select = st.sidebar.radio(
    "Input Source", ["Webcam", "Webcamv2", "Image"]
)

#Select the model to be used
model_select = st.sidebar.radio("Select Model", sorted(list(models.keys())))


model = load_model(model_select)

#Use a 2-row display instead of a sidebar
display_columns = st.columns(2)
with display_columns[0]:
    st.header("Options")

    st.text(f"Loaded model {model_select}")

    categories = sorted(model.names.items(), key = lambda x:x[0])
    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.multiselect("Categories", categories, format_func = lambda x: f"{x[0]}: {x[1]}", placeholder = "All classes")
    conf_threshold = float(st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01))
    iou_threshold = float(st.slider("IoU Threshold", 0.0, 1.0, 0.1, 0.01))

with display_columns[1]:
    match source_select:
        case "Webcam":
            infer_stream = webrtc_streamer(
                key = "Inference Result", 
                mode = WebRtcMode.SENDRECV, 
                rtc_configuration = COMMON_RTC_CONFIG, 
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
                        infer_stream.video_processor.conf_threshold = conf_threshold
                        infer_stream.video_processor.iou_threshold = iou_threshold
                        infer_stream.video_processor.classes_list = [i[0] for i in selected_classes] if selected_classes else None
                        infer_stream.video_processor.show_image = show_image
                        infer_stream.video_processor.show_boxes = show_boxes
                        infer_stream.video_processor.show_masks = show_masks
                        try:
                            frame_no = infer_stream.video_processor.frame_queue.get(timeout = 1.0)
                            tf = time.time()
                            fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
                            fps_text.write(f"FPS: {fps:.2f}")
                        except queue.Empty:
                            print("No image found.")
                            fps_text.write("FPS: 0.00")
                        else:
                            print("Image found!")
                            ti = tf
                    else:
                        print("Ending inference.")
                        break

        case "Webcamv2":
            infer_stream = webrtc_streamer(
                key = "Inference Result", 
                mode = WebRtcMode.SENDONLY, 
                rtc_configuration = COMMON_RTC_CONFIG, 
                media_stream_constraints = {"video": True, "audio":False}, 
            )

            image_place = st.empty()
            fps_text = st.text("FPS: None")
            fps = 0.0
            ti = time.time()
            while True:
                if infer_stream.video_receiver:
                    try:
                        frame = infer_stream.video_receiver.get_frame(timeout = 1)
                    except queue.Empty:
                        break
                    
                    image = frame.to_ndarray(format = "bgr24")
                    results = model.track(image, conf = conf_threshold, iou = iou_threshold, classes = [i[0] for i in selected_classes] if selected_classes else None)
                    result_frame = plot_results(results, show_image, show_boxes, show_masks)
                    output_frame = av.VideoFrame.from_ndarray(result_frame, format = "bgr24")
                    image_place.image(output_frame)
                    tf = time.time()
                    fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
                    fps_text.write(f"FPS: {fps:.2f}")
                    ti = tf
                else:
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
                results = model.track(image, conf = conf_threshold, iou = iou_threshold, classes = [i[0] for i in selected_classes] if selected_classes else None)
                result_frame = plot_results(results, show_image, show_boxes, show_masks)
                st.text("Results")
                st.image(result_frame, caption = "Result Image", use_column_width = True, channels = "BGR", output_format = "auto")
            else:
                st.text("No Uploaded Image")
                st.text("No Results")


        case _:
            st.error("Error: No input source selected!")
