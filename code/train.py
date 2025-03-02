'''
train.py

Serves to train the CNN model in this project by 
connecting an instance of the CNN to input datsets.

Lukas Zumwalt
3/1/2025
'''
import os
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from baseline import BaselineCNN

# Define dataset path
DATASET_PATH = "dataset/train"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Custom dataset loader
class FlowersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Load dataset
dataset = FlowersDataset(DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN(num_classes=len(dataset.classes)).to(device)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    # Training loop
    epochs = 5
    duration = 0
    for epoch in range(epochs):
        t1 = time.time()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)  # Apply Softmax
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        dt = time.time() - t1   # Duration of epoch
        duration += dt          # Accumulated full duration
        print(f"Epoch {epoch+1} (T = {dt:.2f}s), Loss: {running_loss/len(dataloader)}")

    print("Training complete.")
