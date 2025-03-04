'''
learning_report.py

Batches the train and test scripts in one file
over selected portions of the training data.  This
intends to show how well a model can learn.

Lukas Zumwalt
3/2/2025
'''
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from baseline import BaselineCNN

# Define dataset paths
TRAIN_DATASET_PATH = "dataset/train"
TEST_DATASET_PATH = "dataset/test"

# Hyper Parameters
LEARNING_RATE = 0.05
SGD_LR = 0.05
ADAM_LR = 0.001
BATCH_SIZE = 16
EPOCHS = 20


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
    def __init__(self, root_dir, txform=None, subset_ratio=1.0):
        self.root_dir = root_dir
        self.txform = txform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            img_list = os.listdir(class_dir)
            random.shuffle(img_list)  # Randomize selection
            subset_size = int(len(img_list) * subset_ratio)  # Select portion of data
            for img_name in img_list[:subset_size]:
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


if __name__ == "__main__":
    # Training and testing with different dataset sizes
    subset_ratios = [0.25, 0.50, 0.75, 1.00]
    accuracy_results = []
    T0 = time.time()
    for subset in subset_ratios:
        for i in range(1,3):
            print(f"Training with {subset * 100}% of the dataset...")
            # TRAINING
            # Load dataset
            train_dataset = FlowersDataset(TRAIN_DATASET_PATH, txform=TRANSFORM, subset_ratio=subset)
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = BaselineCNN(num_classes=len(train_dataset.classes)).to(device)

            # Define loss and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            if i == 1:
                optimizer = optim.SGD(model.parameters(), lr=SGD_LR)
                print('+ SGD Optimizer')
                print(f'Learning Rate: {SGD_LR}')
            else:
                optimizer = optim.Adam(model.parameters(), lr=ADAM_LR)
                print('+ Adam Optimizer')
                print(f'Learning Rate: {ADAM_LR}')

            # Reduce LR every 5 epochs by a factor of 0.5
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  

            # Training loop
            duration = 0
            print(f'Total Epochs: {EPOCHS}\n')
            t0 = time.time()
            for epoch in range(EPOCHS):
                t1 = time.time()
                running_loss = 0.0
                for images, labels in train_dataloader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                scheduler.step()  # update learning rate per schedule
                dt = time.time() - t1   # Duration of epoch
                duration += dt          # Accumulated full duration
                print(f"Epoch {epoch+1} (T = {dt:.2f}s), Loss: {running_loss/len(train_dataloader)}")

            # Save off model instance
            print(f"Training complete. Took {time.time()-t0:.2f}s")
            torch.save(model.state_dict(), f"bin/flower_model_{subset*100:.0f}train_{i}.pth")
            print(f"Model saved at: bin/flower_model_{subset}train_{i}.pth")

            # TESTING
            # Load test dataset
            test_dataset = FlowersDataset(TEST_DATASET_PATH, txform=TRANSFORM, subset_ratio=1.0)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Load the trained model
            model.eval()  # Set model to evaluation mode

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
            print(f"Test Accuracy with {subset * 100}% of training data: {accuracy:.2f}%\n")
            accuracy_results.append(accuracy)

    print(f'Trained with {subset*100}% of training data...')
    print(f"Time spent training all models: {(time.time()-T0)/60:.2f} minutes")
    print("Accuracy Results:" +\
        "\n--------------------------------------" +\
        f"\nSGD)  {subset_ratios[0]}: {accuracy_results[0]:.4f}" +\
        f"\nAdam) {subset_ratios[0]}: {accuracy_results[1]:.4f}" +\
        "\n--------------------------------------" +\
        f"\nSGD)  {subset_ratios[1]}: {accuracy_results[2]:.4f}" +\
        f"\nAdam) {subset_ratios[1]}: {accuracy_results[3]:.4f}"
        "\n--------------------------------------" +\
        f"\nSGD)  {subset_ratios[2]}: {accuracy_results[4]:.4f}" +\
        f"\nAdam) {subset_ratios[2]}: {accuracy_results[5]:.4f}" +\
        "\n--------------------------------------" +\
        f"\nSGD)  {subset_ratios[3]}: {accuracy_results[6]:.4f}" +\
        f"\nAdam) {subset_ratios[3]}: {accuracy_results[7]:.4f}")

    results = np.array(accuracy_results)
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot([r * 100 for r in subset_ratios], results[[0,2,4,6]], marker='o', linestyle='-', color='b')
    plt.plot([r * 100 for r in subset_ratios], results[[1,3,5,7]], marker='o', linestyle='-', color='r')
    plt.legend(['SGD, LR = 0.05','Adam, LR = 0.001'])
    plt.xlabel("Percentage of Training Data")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Learning by Exposure")
    plt.grid(True)
    plt.show()
