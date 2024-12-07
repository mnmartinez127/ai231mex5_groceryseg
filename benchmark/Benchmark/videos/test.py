import cv2
src = "IMG_5143.MOV"
dst = "IMG_5143_2.MP4"

cap = cv2.VideoCapture(src)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = max(1,int(cap.get(cv2.CAP_PROP_FPS)))
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
print("width:", width, "height:", height, "fps:", fps)
out = cv2.VideoWriter(dst, fourcc, fps, size)

while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        #results = model.track(frame,conf=args.c,iou=args.u,device=DEVICE,persist=True)
        #result_frame = results[0].plot(boxes=True,masks=True)
        #out.write(cv2.resize(result_frame,size))
        out.write(cv2.resize(frame,size))
cap.release()
out.release()