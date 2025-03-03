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
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from baseline import BaselineCNN

# Define dataset path
DATASET_PATH = "dataset/train"

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Custom dataset loader
class FlowersDataset(Dataset):
    '''
    Dedicated, custom dataset class for the flower data selected.
    Designed to point to the globally-defined paths in this file
    and in data_sort.py.
    '''
    def __init__(self, root_dir, txform=None):
        self.root_dir = root_dir
        self.txform = txform
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

        if self.txform:
            image = self.txform(image)

        return image, label

# Load dataset
dataset = FlowersDataset(DATASET_PATH, txform=TRANSFORM)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN(num_classes=len(dataset.classes)).to(device)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
LEARNING_RATE = 0.01
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":
    # Training loop
    epochs = 20
    duration = 0
    print(f'Total Epochs: {epochs}\nLearning Rate: {LEARNING_RATE}')
    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        dt = time.time() - t1   # Duration of epoch
        duration += dt          # Accumulated full duration
        print(f"Epoch {epoch+1} (T = {dt:.2f}s), Loss: {running_loss/len(dataloader)}")

    print(f"Training complete. Took {time.time()-t0:.2f}s")
    torch.save(model.state_dict(), "bin/flower_model.pth")
    print("Model saved at: bin/flower_model.pth")
