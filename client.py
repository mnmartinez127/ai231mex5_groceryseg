# webcam streaming (on local machine)
import cv2
import requests
import numpy as np
import time
import argparse

#Define initial constants, should not change!
SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
#allows for custom port setting
PORT_NUMBER = 8002
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

MAX_FRAME_RATE = 60

params = {
    'model_name': 'YOLOv11 Segmentation M',
    "model_device": 0,
    "model_mode":"torch",
    "conf":0.7,
    "iou":0.1,
    "classes":[],
    "show_boxes":True,
    "show_masks":True,
}

cap = cv2.VideoCapture(0)  # Open webcam
fps = 0.0
ti = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': img_encoded.tobytes()}
    response = requests.post(SERVER_URL,params=params, files=files)
    segmented_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    tf = time.time()
    fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
    cv2.putText(segmented_frame,f"FPS: {fps:.2f}",(40,40),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255),1,cv2.LINE_AA)
    ti = tf
    cv2.imshow("Segmented Output", segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()