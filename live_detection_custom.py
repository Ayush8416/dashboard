import cv2
import cvlib as cv
import numpy as np
import os
from tensorflow.keras.models import load_model

# ================= LOAD MODEL =================
model = load_model("plant_disease_model.h5")

# ================= LOAD DISEASE CLASSES =================
DATASET_DIR = "PlantVillage"
disease_classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

ANIMALS = ["dog", "cat", "cow", "horse", "sheep", "bird"]

print("System Started | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ======================================================
    # 1️⃣ HUMAN & ANIMAL DETECTION (YOLO)
    # ======================================================
    bbox, labels, conf = cv.detect_common_objects(
        frame, confidence=0.5, model="yolov3-tiny"
    )

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        label = labels[i]

        if label == "person":
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"Human",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        elif label in ANIMALS:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,"Animal",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)

    # ======================================================
    # 2️⃣ PLANT / LEAF DETECTION (ONLY GREEN AREAS)
    # ======================================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ❗ filter small/noise areas
        if area < 5000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        # avoid detecting human face/clothes as plant
        if h < 100 or w < 100:
            continue

        plant_img = frame[y:y+h, x:x+w]

        try:
            plant_img = cv2.resize(plant_img, (224,224))
            plant_img = plant_img / 255.0
            plant_img = np.expand_dims(plant_img, axis=0)

            pred = model.predict(plant_img, verbose=0)
            disease = disease_classes[np.argmax(pred)]
        except:
            disease = "Unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,200,255),2)
        cv2.putText(frame,
                    f"Plant | {disease}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,200,255),2)

    # ======================================================
    cv2.imshow("Smart Agriculture AI System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
