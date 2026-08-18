import copy
import random
from collections import Counter
from pathlib import Path


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)




DATASET_DIR = Path("D:/dataset3")


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320


BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# keep at 0 initially fpr windows
NUM_WORKERS = 0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class RandomPanoramaRoll:
    """
    Circularly shifts an equirectangular panorama horizontally.


    This changes where the panorama seam appears without flipping
    left/right-driving information.
    """


    def __init__(self, probability: float = 0.8):
        self.probability = probability


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image


        width = image.shape[-1]
        shift = random.randint(0, width - 1)


        return torch.roll(image, shifts=shift, dims=-1)




# normalization values used by ImageNet-pretrained torchvision models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]




train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            antialias=True,
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        RandomPanoramaRoll(probability=0.8),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


evaluation_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)




# ---------------------------------------------------------------------
# Dataset and DataLoaders
# ---------------------------------------------------------------------


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




# Verify that all three splits use the same label mapping.
if train_dataset.class_to_idx != validation_dataset.class_to_idx:
    raise ValueError("Train and validation class mappings do not match.")


if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("Train and test class mappings do not match.")




class_names = train_dataset.classes
number_of_classes = len(class_names)


print(f"Device: {DEVICE}")
print(f"Classes: {class_names}")
print(f"Class mapping: {train_dataset.class_to_idx}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Test images: {len(test_dataset)}")




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




# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------


# These help when one country has substantially more images than another.
class_counts = Counter(train_dataset.targets)


counts_tensor = torch.tensor(
    [class_counts[index] for index in range(number_of_classes)],
    dtype=torch.float32,
)


class_weights = counts_tensor.sum() / (
    number_of_classes * counts_tensor
)


print("Training class counts:")


for class_name, class_index in train_dataset.class_to_idx.items():
    print(f"  {class_name}: {class_counts[class_index]}")


print(f"Class weights: {class_weights.tolist()}")




# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)

input_features = model.classifier[2].in_features

model.classifier[2] = nn.Linear(
    input_features,
    number_of_classes,
)

model = model.to(DEVICE)




# CrossEntropyLoss expects raw model logits and integer class labels.
# Do not apply softmax inside the model before calculating this loss.
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE),
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)




# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    training: bool,
) -> tuple[float, float]:
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




best_validation_accuracy = 0.0
best_model_state = copy.deepcopy(model.state_dict())




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
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"LR: {current_lr:.2e} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation loss: {validation_loss:.4f} | "
        f"Validation accuracy: {validation_accuracy:.2%}"
    )


    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_model_state = copy.deepcopy(model.state_dict())


        torch.save(
            {
                "model_state_dict": best_model_state,
                "class_names": class_names,
                "class_to_idx": train_dataset.class_to_idx,
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "model_name": "convnext_tiny",
            },
            "best_convnext_tiny_V3.pt",
        )


        print("  Saved new best model.")




# ---------------------------------------------------------------------
# Final test evaluation
# ---------------------------------------------------------------------


model.load_state_dict(best_model_state)


test_loss, test_accuracy = run_epoch(
    model,
    test_loader,
    training=False,
)


print()
print(f"Best validation accuracy: {best_validation_accuracy:.2%}")
print(f"Final test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2%}")
print("Saved model: best_convnext_tiny_V3.pt")

