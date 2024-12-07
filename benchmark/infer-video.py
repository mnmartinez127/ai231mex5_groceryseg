import os
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import shutil
import logging
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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


MODELS_DIR = os.path.join(os.getcwd(), "../models")
INPUT_DIR = os.path.join(os.getcwd(),"Benchmark")
OUTPUT_DIR = os.path.join(os.getcwd(),"Benchmark_results")
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LIST = list(range(torch.cuda.device_count()))+["cpu"]
MAX_FRAME_RATE = 30
MODE = "all"
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.01

parser = argparse.ArgumentParser(description="Specify the model(s) and image/video directories to process.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d",default=MODELS_DIR,help="Model Directory")
parser.add_argument("-i",default=INPUT_DIR,help="Input Directory")
parser.add_argument("-o",default=OUTPUT_DIR,help="Output Directory")
parser.add_argument("-m",default=MODE,choices=("best","last","all"),help="Mode List (best|last|all)")
parser.add_argument("-c",default=CONF_THRESHOLD,help="Confidence Threshold")
parser.add_argument("-u",default=IOU_THRESHOLD,help="IoU Threshold")
parser.add_argument("-r",action="store_true",help="Resize Output")
args = parser.parse_args()

if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

session_data = {}
model = {}



model_dict = {}
for model_dir in os.listdir(args.d):
    if os.path.join(args.d, model_dir).endswith(".pt"):
        model_dict[os.path.splitext(model_dir)[0]] = os.path.join(args.d, model_dir)
    elif os.path.isdir(os.path.join(args.d, model_dir)):
        weight_path = os.path.join(args.d, model_dir, "weights", "best.pt")
        if os.path.isfile(weight_path) and args.m in ["best","all"]:
            model_dict[model_dir+"-Best"] = weight_path
        weight_path = os.path.join(args.d, model_dir, "weights", "last.pt")
        if os.path.isfile(weight_path) and args.m in ["last","all"]:
            model_dict[model_dir+"-Last"] = weight_path



session_data[0] = {
    "device":0 if torch.cuda.is_available() else "cpu",
    "conf": args.c, #must be from 0 to 1 inclusive
    "iou": args.u, #must be from 0 to 1 inclusive
    "show-boxes": True,
    "show-masks": True,
    "zoom-scale":1.0, #Scale must be >= 1.0. Disable zoom by setting this to 1.0
    "zoom-mode":"center", #Should be "center" or "grid",otherwise no zoom
    "rescale": False, #only has an effect if zoom is enabled
    "resize": args.r, #resize the image to 640*(h*640/w) before inference
    "cx":0.5, #must be within 0 to 1 exclusive, scaled to image width
    "cy":0.5, #must be within 0 to 1 exclusive, scaled to image height
}
logger.info("Initialized variables!")
logger.info("Model dict: %s",model_dict)
logger.info("Current parameters: %s",session_data[0])



for k,v in model_dict.items():
    print(f"{k}: {v}")

inputs = []

for root, dirs, files in os.walk(args.i):
    for name in files:
        inputs.append((root, name))


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
        results[0].update(boxes=boxes,masks=new_masks) #update rescaled boxes and masks
    else:
        results = model[0].track(image,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)
    #Use 2x zoom if no objects were found
    if not results[0].boxes:
        logger.info("No objects found.")
        if session_data[session_token]["zoom-scale"] > 1.0:
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
                results2[0].update(boxes=boxes,masks=new_masks) #update rescaled boxes and masks
            else:
                results2 = model.get(1,model[0]).track(image2,conf=session_data[session_token]["conf"],iou=session_data[session_token]["iou"],device=session_data[session_token]["device"],persist=persist)

            if session_data[session_token]["rescale"]:
                results2[0].orig_img = results[0].orig_img
                w,h = image.shape[1],image.shape[0]
                left_bound,top_bound,right_bound,bottom_bound,cw,ch,_,_ = get_box(w,h,session_token)
                #Remove this line in production! - CONFIRM ZOOM POSITION
                results2[0].orig_img = cv2.rectangle(results2[0].orig_img,(int(max(0,left_bound)),int(max(0,top_bound))),(int(min(right_bound,w)),int(min(bottom_bound,h))),(0,255,0))
                results2[0].orig_img = cv2.circle(results2[0].orig_img,(int(cw),int(ch)),5,(0,255,0),2)
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


def infer_media(model,src,dst,resize_output=False,session_token=0):
    #Image input
    if os.path.splitext(src)[1].lower() in [".jpg",".jpeg",".png",]:
        image = cv2.imread(src)
        result_frame = infer_image(image,persist=False)
        cv2.imwrite(dst,result_frame)
    #Video input
    if os.path.splitext(src)[1].lower() in [".mp4",".mkv",".mov",]:
        dst_folder = os.path.splitext(dst)[0]
        os.makedirs(dst_folder,exist_ok=True)
        frame_ctr = 0
        cap = cv2.VideoCapture(src)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fps = max(1,int(cap.get(cv2.CAP_PROP_FPS)))
        size = (width, height)
        print("width:", width, "height:", height, "fps:", fps)
        persist = False #tripped on first inference
        while cap.isOpened():
            ret,frame = cap.read()
            if ret:
                result_frame = infer_image(frame,persist=persist)
                persist = True #track stream for succeeding images
                cv2.imshow(k,result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                print("SAVE TO: ",os.path.join(dst_folder,f"{frame_ctr}.jpg"))
                cv2.imwrite(os.path.join(dst_folder+"_raw",f"{frame_ctr}.jpg"),frame)
                cv2.imwrite(os.path.join(dst_folder,f"{frame_ctr}.jpg"),result_frame)
                frame_ctr += 1
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

torch.cuda.empty_cache()
for k,v in model_dict.items():
    if "(Augmented)" in k:
        model[0] = YOLO(v,task="segment")
        model[1] = YOLO(v,task="segment")
        print(f"Model: {k}")
        for root, name in inputs:
            src = os.path.join(root,name)
            dst = os.path.join(args.o,k,os.path.relpath(src,args.i))
            print(f"Reading: {src}")
            print(f"Writing: {dst}")
            os.makedirs(os.path.split(dst)[0],exist_ok=True)
            if not os.path.isfile(dst):
                infer_media(model,src,dst,args.r)
        model[0].data = None
        model[1].data = None
        model.clear()
        torch.cuda.empty_cache()