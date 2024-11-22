# webcam streaming (on local machine)
import cv2
import requests
import numpy as np

import sys
if len(sys.argv) > 1:
    PORT_NUMBER = int(sys.argv[1])
else:
    PORT_NUMBER = 8003

# Define the server endpoint
url = f"http://202.92.159.241:{PORT_NUMBER}/infer_frame"
#url = f"http://127.0.0.1:{PORT_NUMBER}/infer_frame"

MAX_FRAME_RATE = 60


cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': img_encoded.tobytes()}
    response = requests.post(url, files=files)
    segmented_frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow("Segmented Output", segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()