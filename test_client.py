# webcam streaming (on local machine)
import cv2
import requests
import numpy as np
import sys

#Define initial constants, should not change!
SERVER_ADDRESS = "http://202.92.159.241"   #DGX server 1
#SERVER_ADDRESS = "http://202.92.159.242"   #DGX server 2
#SERVER_ADDRESS = "http://127.0.0.1"        #Localhost, inference server
#allows for custom port setting
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8003
PAGE = "/infer_frame"
SERVER_URL = f"{SERVER_ADDRESS}:{PORT_NUMBER}{PAGE}"

MAX_FRAME_RATE = 60


cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': img_encoded.tobytes()}
    response = requests.post(SERVER_URL, files=files)
    segmented_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow("Segmented Output", segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()