import cv2
import cvlib as cv
import numpy as np

# # Corrected file paths
# prototxt_path = '/Users/hwan/Documents/ml/003_YOLO_computer_vision/02_face_detection_more_than/gender_deploy.prototxt'
# model_path = '/Users/hwan/Documents/ml/003_YOLO_computer_vision/02_face_detection_more_than/gender_net.caffemodel'
#
# # Load the model
# net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture("people2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 500))

    faces, confidences = cv.detect_face(frame)
    for face, conf in zip(faces, confidences):
        x, y = face[0], face[1]
        x1, y1 = face[2], face[3]
        crop = frame[y:y1, x:x1]
        # (label, confidence) = cv.detect_gender(crop)
        # idx = np.argmax(confidence)
        # label = label[idx]
        # print(label)

        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        # cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()