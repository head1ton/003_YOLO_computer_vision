import cv2
import numpy as np
import torch
from tracker import Tracker

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture('cctv.mp4')

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse position: {x}, {y}")

cv2.namedWindow("FRAME")
cv2.setMouseCallback('FRAME', POINTS)

area_1 = [(377, 315), (429, 373), (535, 339), (500, 296)]
area1 = set()

tracker = Tracker()
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))

    cv2.polylines(frame, [np.array(area_1, np.int32)], True, (0,255,0), 3)

    results = model(frame)
    # frame = np.squeeze(results.render())
    # a = results.pandas().xyxy[0]
    # print(a)
    list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        # print(row)
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        b = str(row['name'])
        if 'person' in b:
            list.append([x1, y1, x2, y2])
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # cv2.putText(frame, b, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    boxes_ids = tracker.update(list)
    # print(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        result = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(w), int(h)), False)
        if result >= 0:
            area1.add(id)

    # print(area1, len(area1))
    p = len(area1)
    cv2.putText(frame, str(p), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()