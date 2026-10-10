import cv2
import mediapipe as mp
import os
import numpy as np
import csv

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------------
# MediaPipe Hand Landmarker
# -----------------------------------

base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------------
# Normalize landmarks
# -----------------------------------

def normalize_landmarks(hand_landmarks):

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Wrist becomes origin
    points = points - points[0]

    # Scale by hand size
    scale = np.max(
        np.linalg.norm(points, axis=1)
    )

    if scale > 0:
        points = points / scale

    return points.flatten()


# -----------------------------------
# Dataset location
# -----------------------------------

dataset_path = (
    "dataset/asl_alphabet_train/"
    "asl_alphabet_train"
)

output_file = "landmark_data.csv"


# -----------------------------------
# CSV header
# -----------------------------------

header = [f"feature_{i}" for i in range(63)]
header.append("label")


total_images = 0
detected_images = 0


# -----------------------------------
# Extract landmarks
# -----------------------------------

with open(
    output_file,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(header)

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

        folder = os.path.join(
            dataset_path,
            letter
        )

        images = os.listdir(folder)

        print(
            f"\nProcessing {letter}: "
            f"{len(images)} images"
        )

        for index, filename in enumerate(images):

            total_images += 1

            path = os.path.join(
                folder,
                filename
            )

            image = cv2.imread(path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            features = normalize_landmarks(
                result.hand_landmarks[0]
            )

            row = features.tolist()
            row.append(letter)

            writer.writerow(row)

            detected_images += 1

            # Progress every 500 images
            if (index + 1) % 500 == 0:

                print(
                    f"  {index + 1}/{len(images)} "
                    f"processed"
                )


print("\n==============================")
print("Extraction complete!")
print("==============================")
print("Total images:", total_images)
print("Hands detected:", detected_images)
print("Images skipped:", total_images - detected_images)
print("Saved to:", output_file)

detector.close()