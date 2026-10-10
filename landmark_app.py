import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load trained model and label encoder
model = joblib.load("models/landmark_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")


# Load MediaPipe hand detector
base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# Normalize landmarks as during training
def normalize_landmarks(hand_landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Make the wrist the origin
    points = points - points[0]

    # Normalize for hand size
    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 0:
        points = points / scale

    return points.flatten().reshape(1, -1)


# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    detector.close()
    raise SystemExit


print("ASL Landmark Recognition started!")
print("Show one ASL hand sign.")
print("Press Q to quit.")

timestamp_ms = 0
prediction_history = deque(maxlen=7)

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        # Mirror webcam view
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms += 33

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Draw hand landmarks and connections
            height, width = frame.shape[:2]

            points = [
                (int(lm.x * width), int(lm.y * height))
                for lm in hand
            ]

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]

            for start, end in connections:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            for point in points:
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 255, 255),
                    -1
                )

            # Predict from the 63 hand landmark values
            features = normalize_landmarks(hand)

            predicted_id = model.predict(features)[0]
            prediction_history.append(predicted_id)

            # Smooth predictions using the most frequent recent result
            stable_id = max(
                set(prediction_history),
                key=prediction_history.count
            )

            letter = label_encoder.inverse_transform(
                [stable_id]
            )[0]

            # Display predicted letter
            cv2.putText(
                frame,
                f"Letter: {letter}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Display model score
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = float(np.max(probabilities)) * 100

                cv2.putText(
                    frame,
                    f"Model score: {confidence:.1f}%",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        else:
            # Clear old predictions when no hand is visible
            prediction_history.clear()

            cv2.putText(
                frame,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("ASL Landmark Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    detector.close()
