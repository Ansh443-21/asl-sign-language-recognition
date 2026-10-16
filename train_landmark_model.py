
import os
import cv2
import mediapipe as mp
import numpy as np
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = "asl_alphabet_train/asl_alphabet_train"
MODEL_PATH = "models/landmark_model.joblib"
ENCODER_PATH = "models/label_encoder.joblib"

os.makedirs("models", exist_ok=True)

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

def normalize_landmarks(hand):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )
    points -= points[0]
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points /= scale
    return points.flatten()

X, y = [], []
total = 0
detected = 0

print("Starting landmark extraction...")
print("This can take a while. Please keep this terminal open.")

with vision.HandLandmarker.create_from_options(options) as detector:
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        folder = os.path.join(DATASET_PATH, letter)
        if not os.path.isdir(folder):
            print("Missing folder:", folder)
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"\n{letter}: {len(files)} images")

        for i, filename in enumerate(files):
            total += 1
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                X.append(normalize_landmarks(result.hand_landmarks[0]))
                y.append(letter)
                detected += 1

            if (i + 1) % 500 == 0:
                print(
                    f"  {i + 1}/{len(files)} processed; "
                    f"hands detected: {detected}"
                )

print(f"\nProcessed {total} images; detected hands in {detected}.")

if len(X) == 0:
    raise RuntimeError(
        "No hands detected. Check the dataset and hand detector."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Validation accuracy on extracted samples: {accuracy:.2%}")

joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print("\nSaved:", MODEL_PATH)
print("Saved:", ENCODER_PATH)
print("Training complete!")