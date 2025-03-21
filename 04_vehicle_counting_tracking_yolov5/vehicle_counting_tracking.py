import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tracker import Tracker

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = YOLO('yolov5s.pt')

cap = cv2.VideoCapture("highway.mp4")

count = 0
tracker = Tracker()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse position: {x}, {y}")

cv2.namedWindow("FRAME")
cv2.setMouseCallback("FRAME", POINTS)

area1 = [(448, 445), (428, 472), (526, 482), (524, 449)]
area2 = [(710, 429), (724, 442), (775, 434), (769, 419)]
area_1 = set()
area_2 = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)

    list = []
    for index, rows in results.pandas().xyxy[0].iterrows():
        x = int(rows['xmin'])
        y = int(rows['ymin'])
        x1 = int(rows['xmax'])
        y1 = int(rows['ymax'])
        b = str(rows['name'])
        list.append([x, y, x1, y1])
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        # cv2.putText(frame, b, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    idx_bbox = tracker.update(list)

    for bbox in idx_bbox:
        x2, y2, x3, y3, id = bbox

        cv2.rectangle(frame, (x2, y2), (x3, y3), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x2, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(frame,  (x3, y3), 4, (0, 255, 0), -1)
        result = cv2.pointPolygonTest(np.array(area1, np.int32), (x3, y3), False)
        result1 = cv2.pointPolygonTest(np.array(area2, np.int32), (x3, y3), False)
        if result > 0:
            area_1.add(id)
        if result1 > 0:
            area_2.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 255), 3)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 255), 3)
    a1 = len(area_1)
    a2 = len(area_2)

    cv2.putText(frame, str(a1), (549, 459), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    cv2.putText(frame, str(a2), (804, 411), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)


    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()