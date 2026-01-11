"""
AI-Based Transit Fare Validator
--------------------------------
This system detects a human face from live video,
predicts the age using a pretrained deep learning model,
and validates eligibility for child and senior citizen fare concessions.
"""

import cv2
import numpy as np

# -------------------------------
# Configuration & Model Paths
# -------------------------------
FACE_PROTO = "model/deploy.prototxt"
FACE_MODEL = "model/res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "model/age_deploy.prototxt"
AGE_MODEL = "model/age_net.caffemodel"

CONFIDENCE_THRESHOLD = 0.7

AGE_BUCKETS = [
    '(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)',
    '(33-43)', '(44-53)', '(60-100)'
]

# -------------------------------
# Load Pretrained Models
# -------------------------------
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# -------------------------------
# Helper Functions
# -------------------------------
def get_fare_status(age_group: str):
    """
    Determines fare eligibility based on predicted age group.
    Returns status message, frame color, age text color, status text color.
    """
    if age_group in ['(0-2)', '(4-6)', '(8-12)']:
        return "Child Discount Allowed", (0, 255, 100),(255, 255, 0),  (0, 255, 100)
    elif age_group == '(60-100)':
        return "Senior Discount Allowed", (0, 255, 100),(255, 255, 0),  (0, 255, 100) 
    else:
        return "No Discount", (0,0,255), (0,255,255), (0, 0, 255)


def detect_faces(frame, width, height):
    """
    Detects faces in a frame using OpenCV DNN face detector.
    """
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (350, 350)),
        1.0,
        (350, 350),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    return face_net.forward()


def predict_age(face):
    """
    Predicts age group from a detected face.
    """
    face_blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )

    age_net.setInput(face_blob)
    predictions = age_net.forward()
    return AGE_BUCKETS[predictions[0].argmax()]

# -------------------------------
# Main Execution
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)

    print("AI-Based Transit Fare Validator Started...")
    print("Press 'Q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        detections = detect_faces(frame, width, height)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                x1, y1, x2, y2 = box.astype(int)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                age_group = predict_age(face)
                status, frame_color, age_color, status_color = get_fare_status(age_group)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)
                cv2.putText(
                    frame,
                    f"Age Group: {age_group}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    age_color,
                    2
                )
                cv2.putText(
                    frame,
                    status,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2
                )

        cv2.imshow("AI Transit Fare Validator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main()
