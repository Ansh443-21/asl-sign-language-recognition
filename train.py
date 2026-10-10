import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


# =========================
# 1. Basic settings
# =========================

DATASET_PATH = "dataset/asl_alphabet_train/asl_alphabet_train"
MODEL_PATH = "models/best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# 2. Image transformations
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(p=0.0),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 3. Load dataset
# =========================

full_dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)


# =========================
# 4. Keep only A-Z
# =========================

allowed_classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

selected_indices = [
    i for i, (_, label) in enumerate(full_dataset.samples)
    if full_dataset.classes[label] in allowed_classes
]

filtered_samples = [
    full_dataset.samples[i]
    for i in selected_indices
]

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(allowed_classes)
}

class_names = allowed_classes

print("Number of A-Z images:", len(filtered_samples))


# =========================
# 5. Create train/validation split
# =========================

dataset_size = len(filtered_samples)

train_size = int(0.85 * dataset_size)
validation_size = dataset_size - train_size

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()

train_indices = indices[:train_size]
validation_indices = indices[train_size:]


class ASLDataset(torch.utils.data.Dataset):

    def __init__(self, samples, indices, transform):
        self.samples = samples
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        sample_index = self.indices[index]

        image_path, original_label = self.samples[sample_index]

        image = datasets.folder.default_loader(image_path)

        class_name = full_dataset.classes[original_label]

        label = class_to_idx[class_name]

        image = self.transform(image)

        return image, label


train_dataset = ASLDataset(
    filtered_samples,
    train_indices,
    train_transform
)

validation_dataset = ASLDataset(
    filtered_samples,
    validation_indices,
    validation_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# =========================
# 6. Load EfficientNet-B0
# =========================

print("Loading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Replace the final classification layer
number_of_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    number_of_features,
    26
)

model = model.to(device)


# =========================
# 7. Loss and optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)


# =========================
# 8. Training
# =========================

best_validation_accuracy = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # ---------------------
    # Training
    # ---------------------

    model.train()

    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = 100 * correct / total

    # ---------------------
    # Validation
    # ---------------------

    model.eval()

    correct = 0
    total = 0
    validation_loss = 0.0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    validation_accuracy = 100 * correct / total

    print(
        f"Train Loss: {running_loss / len(train_loader):.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{validation_loss / len(validation_loader):.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )


    # ---------------------
    # Save best model
    # ---------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": class_names
            },
            MODEL_PATH
        )

        print("Best model saved!")


print()
print("Training complete!")
print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2f}%"
)