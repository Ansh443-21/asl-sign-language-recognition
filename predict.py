import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Classes
# -----------------------------
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# -----------------------------
# Create EfficientNet-B0
# -----------------------------
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    26
)


# -----------------------------
# Load trained model
# -----------------------------
model_path = "models/best_model.pth"

checkpoint = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image):
    """
    Takes a PIL image and returns:
    predicted letter and confidence.
    """

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    letter = classes[predicted.item()]
    confidence = confidence.item() * 100

    return letter, confidence