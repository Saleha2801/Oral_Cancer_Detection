import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Histopathology Model on device: {device}")

# Dataset Class
class HistopathologyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # 0: OSCC (Cancer), 1: Leukoplakia (Non-cancer)
        label = 0 if row["TaskII"] == "OSCC" else 1
        return image, label

train_csv = "NDU-UFES/histopathological img dataset/ndbufes_TaskII_parsed_folders.csv"
test_csv = "NDU-UFES/histopathological img dataset/ndbufes_TaskII_parsed_test.csv"
img_dir = "NDU-UFES/histopathological images"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader = DataLoader(HistopathologyDataset(train_df, img_dir, train_transform), batch_size=16, shuffle=True)
test_loader = DataLoader(HistopathologyDataset(test_df, img_dir, val_transform), batch_size=16, shuffle=False)

# Build ResNet-18
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("Loaded ImageNet pretrained weights.")
except Exception as e:
    print(f"Loading local resnet18 initialized without weights: {e}")
    model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

best_acc = 0.0
best_state = None
num_epochs = 8

t0 = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    epoch_loss = running_loss / len(train_df)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Test Acc: {acc:.4f}")
    if acc >= best_acc:
        best_acc = acc
        best_state = model.state_dict()

print(f"\nTraining completed in {time.time()-t0:.1f}s. Best Test Accuracy: {best_acc:.4f}")

# Save best model
os.makedirs("models", exist_ok=True)
out_path = "models/best_histopathology_model.pth"
torch.save(best_state, out_path)
print(f"Saved best histopathology model to {out_path}")
