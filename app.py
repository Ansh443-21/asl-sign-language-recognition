import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load MediaPipe hand landmark model
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


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

print("Hand landmark detector started.")
print("Show your hand to the camera.")
print("Press Q to quit.")


timestamp = 0

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 1

    # Detect hand landmarks
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Draw the 21 landmarks
    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            h, w, _ = frame.shape

            points = []

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

    cv2.putText(
        frame,
        "21-Point Hand Landmark Detection",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "ASL Hand Landmarks",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
detector.close()