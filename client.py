# webcam streaming (on local machine)
import cv2
import requests
import numpy as np
import time

import sys
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8003

# Define the server endpoint
url = f"http://202.92.159.241:{PORT_NUMBER}/infer_frame"
#url = f"http://127.0.0.1:{PORT_NUMBER}/infer_frame"

MAX_FRAME_RATE = 60

params={
    'model_name': 'YOLOv11 Segmentation M',
    'conf': 0.5,
    'iou': 0.1,
    'classes': None,
    'show_boxes': True,
    'show_masks': True,
    'mode':'torch',
    'device':0,
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
    response = requests.post(url,params=params, files=files)
    segmented_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    tf = time.time()
    fps = 0.0 if (tf-ti == 0.0) else max(1.0/(tf-ti), 1.0/MAX_FRAME_RATE)
    cv2.putText(segmented_frame,f"FPS: {fps:.2f}",(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2,cv2.LINE_AA)
    ti = tf
    cv2.imshow("Segmented Output", segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()