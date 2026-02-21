import os
import json
import random
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# ===========================
# CONFIG
# ===========================
IMAGE_DIR = "./images"
ANNOTATION_DIR = "./annotations"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# STEP 1: FIND TOP-5 CATEGORIES
# ===========================
def find_top5_categories(annotation_dir):
    counter = Counter()

    for file in os.listdir(annotation_dir):
        if file.endswith(".json"):
            with open(os.path.join(annotation_dir, file)) as f:
                data = json.load(f)
                for key in data:
                    if key.startswith("item"):
                        counter[data[key]["category_id"]] += 1

    top5 = [x[0] for x in counter.most_common(5)]
    return top5


# ===========================
# DATASET
# ===========================
class DeepFashionMultiLabel(Dataset):
    def __init__(self, image_files, image_dir, annotation_dir, top5_ids, transform=None):
        self.image_files = image_files
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.top5_ids = top5_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".json"))

        image = Image.open(img_path).convert("RGB")

        multi_label = torch.zeros(len(self.top5_ids))

        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)
                for key in data:
                    if key.startswith("item"):
                        cat_id = data[key]["category_id"]
                        if cat_id in self.top5_ids:
                            idx_label = self.top5_ids.index(cat_id)
                            multi_label[idx_label] = 1

        if self.transform:
            image = self.transform(image)

        return image, multi_label


# ===========================
# SPLIT DATA
# ===========================
def split_data(image_dir, split_ratio=0.8):
    files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    random.shuffle(files)

    split = int(len(files) * split_ratio)
    return files[:split], files[split:]


# ===========================
# MODEL
# ===========================
def get_model(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===========================
# TRAINING
# ===========================
def train():
    print("Finding top 5 categories...")
    top5_ids = find_top5_categories(ANNOTATION_DIR)
    print("Top 5 Category IDs:", top5_ids)

    train_files, val_files = split_data(IMAGE_DIR)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = DeepFashionMultiLabel(
        train_files, IMAGE_DIR, ANNOTATION_DIR, top5_ids, transform
    )

    val_dataset = DeepFashionMultiLabel(
        val_files, IMAGE_DIR, ANNOTATION_DIR, top5_ids, transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = get_model(num_classes=5).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Micro F1: {micro_f1:.4f}")
        print("------------------------------------------------")

    torch.save(model.state_dict(), "resnet50_multilabel.pth")
    print("Model saved as resnet50_multilabel.pth")


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    train()
