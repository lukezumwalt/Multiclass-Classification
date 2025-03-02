'''
test.py

Accepts and tests a dedicated data set.

Depends upon existing model to have been trained
separately and saved by train.py.

May be improved in future...

Lukas Zumwalt
3/2/2025
'''
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from baseline import BaselineCNN  # Import the trained model

# Define test dataset path
TEST_DATASET_PATH = "dataset/test"

# Image preprocessing (same as train.py)
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Custom dataset loader for test data
class TestDataset(Dataset):
    '''
    ! Potentially redundant !
    Test Dataset class for handling the split of test data.
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

# Load test dataset
test_dataset = TestDataset(TEST_DATASET_PATH, txform=TRANSFORM)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # setup torch device
model = BaselineCNN(num_classes=len(test_dataset.classes)).to(device) # define empty model
# overwrite this model instance with the learned pars:
model.load_state_dict(torch.load("bin/flower_model.pth"))
model.eval()  # Set model to evaluation mode

if __name__ == "__main__":
    # Evaluate the model
    correct = 0
    total = 0

    with torch.no_grad():
        print(f"Batch size: {test_loader.batch_size}")
        for images, labels in tqdm(test_loader, desc="Testing Progress", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
