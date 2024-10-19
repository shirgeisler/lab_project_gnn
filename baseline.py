import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 30
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Step 4: Use ImageFolder with the selected classes
def filter_dataset(dataset, selected_classes):
    """Filters an ImageFolder dataset to include only the selected classes and remaps class indices."""
    # Get index of selected classes based on original dataset's class_to_idx
    selected_class_idxs = [dataset.class_to_idx[cls] for cls in selected_classes]

    # Filter the samples (tuple of (path, class_idx)) to include only the selected class indices
    filtered_samples = [sample for sample in dataset.samples if sample[1] in selected_class_idxs]

    # Remap class indices to be in the range [0, len(selected_classes) - 1]
    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_class_idxs)}
    remapped_samples = [(path, idx_mapping[class_idx]) for (path, class_idx) in filtered_samples]

    # Update dataset
    dataset.samples = remapped_samples
    dataset.targets = [target for _, target in remapped_samples]

    # Update the class_to_idx and classes attributes to reflect only the selected classes
    dataset.class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    dataset.classes = selected_classes

    return dataset


def get_loaders():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    train_path = 'data/tiny-imagenet-200/train'
    test_path = 'data/tiny-imagenet-200/val'
    all_classes = os.listdir(train_path)  # Each sub-folder represents a class
    selected_classes = random.sample(all_classes, 10)
    print(f"Selected Classes: {selected_classes}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    # Step 5: Filter the datasets to keep only the selected classes
    train_dataset = filter_dataset(train_dataset, selected_classes)
    test_dataset = filter_dataset(test_dataset, selected_classes)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader


# Baseline Models: ResNet-18, VGG-16, MobileNetV2
def get_model(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated to use `weights`
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)  # Updated to use `weights`
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)  # Updated to use `weights`
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model.to(device)


# Train function
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {correct / len(loader.dataset):.4f}")


# Test function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {correct / len(loader.dataset):.4f}")


if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 10

    # Main training loop for each baseline
    for model_name in [
                       #'resnet18',
                       'mobilenet_v2']:
        print(f"\nTraining {model_name}...")
        model = get_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            train(model, train_loader, criterion, optimizer)
            evaluate(model, test_loader, criterion)
