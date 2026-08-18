import copy
import random
from collections import Counter
from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)

with open("config.json", "r") as file:
    config = json.load(file)

DATASET_DIR = Path(config["outputDirectory"])
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = .0001
WEIGHT_DECAY = .0001

# keep at 0 initially for windows
NUM_WORKERS = 0

# use cuda if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# roll the panorama around to virtually change direction the camera is pointed in
# this effectively changes where the seam in the image is and keeps left/right hand driving the same
class RandomPanoramaRoll:

    def __init__(self, probability: float = 0.8):
        self.probability = probability

    # roll the image 80% of the time
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image

        width = image.shape[-1]
        shift = random.randint(0, width - 1)

        return torch.roll(image, shifts=shift, dims=-1)


# normalization values used by the ImageNet pretrained torchvision models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose(
    [
        # jitter and change lighting and color because we are really after classifying geographic features
        # and not just geoguessr specific artifacts
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(), # convert pixel values to pytorch tensor
        RandomPanoramaRoll(probability=0.8),
        # normalize to make images look more like the images the pretrained model was trained on
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# dont augment the validation and test data
evaluation_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# dataset
train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform,
)

validation_dataset = datasets.ImageFolder(
    DATASET_DIR / "validation",
    transform=evaluation_transform,
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=evaluation_transform,
)


# ensure that all the splits mapped the same
if train_dataset.class_to_idx != validation_dataset.class_to_idx:
    raise ValueError("train and validation class mappings do not match")

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("train and test class mappings do not match")


classNames = train_dataset.classes

print(f"device: {DEVICE}")
print(f"classes: {len(classNames)}")
print(f"class mapping: {train_dataset.class_to_idx}")
print(f"training images: {len(train_dataset)}")
print(f"validation images: {len(validation_dataset)}")
print(f"test images: {len(test_dataset)}")


# data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE.type == "cuda",
)

# class weights for when image distribution isnt uniform
class_counts = Counter(train_dataset.targets)
numClasses = len(classNames)

counts_tensor = torch.tensor(
    [class_counts[index] for index in range(numClasses)],
    dtype=torch.float32,
)

# give rarer classes more importance
class_weights = counts_tensor.sum() / (
    numClasses * counts_tensor
)

print("Training class counts:")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(f"  {class_name}: {class_counts[class_index]}")


print(f"Class weights: {class_weights.tolist()}")


# model

# downloads ImageNet pretrained ConvNeXt-Tiny
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

# replace finals layer with one that has one class per country
input_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(
    input_features,
    numClasses,
)

model = model.to(DEVICE)

# cross entropy loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE),
)
 # AdamW optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# drops learning rate when progress slows for microadjustments
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

# training and eval
def run_epoch(model: nn.Module, loader: DataLoader, training: bool) -> tuple[float, float]:
    if training:
        model.train()
    else:
        model.eval()


    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy

# store bestmodel based on validation accuracy
best_validation_accuracy = 0.0
best_model_state = copy.deepcopy(model.state_dict())

# outer training loop
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_accuracy = run_epoch(
        model,
        train_loader,
        training=True,
    )

    validation_loss, validation_accuracy = run_epoch(
        model,
        validation_loader,
        training=False,
    )

    scheduler.step(validation_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        f"LR: {current_lr:.2e}"
        f"Train loss: {train_loss:.4f}"
        f"Train accuracy: {train_accuracy:.2%}"
        f"Validation loss: {validation_loss:.4f}"
        f"Validation accuracy: {validation_accuracy:.2%}"
    )

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model_state = copy.deepcopy(model.state_dict())

        torch.save(
            {
                "model_state_dict": best_model_state,
                "class_names": classNames,
                "class_to_idx": train_dataset.class_to_idx,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "model_name": "convnext_tiny",
            },
            "best_convnext_tiny.pt",
        )

        print("\nSaved new best model.")


# test eval
model.load_state_dict(best_model_state)
test_loss, test_accuracy = run_epoch(model, test_loader, training=False,)

print(f"\nBest validation accuracy: {best_validation_accuracy:.2%}")
print(f"Final test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2%}")
print("Saved model: best_convnext_tiny.pt")