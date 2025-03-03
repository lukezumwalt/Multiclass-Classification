'''
data_sort.py

Sorts bulk data into a 70% training and 30% testing split.
Places images into respective new directories.

Custom designed to handle the Alex Mamaev kaggle flowers
after manually downloading and extracting.  BASE_DIR may
be updated if a different data set is desired, and 
CATEGORIES would need to be modified too.

Every time this script is ran

Lukas Zumwalt
3/1/2025
'''
import os
import shutil
import random

# Define the paths
BASE_DIR = 'dataset/archive/flowers'  # Root directory containing subfolders of images
# TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TRAIN_DIR = 'dataset/train'
# TEST_DIR = os.path.join(BASE_DIR, 'test')
TEST_DIR = 'dataset/test'

# Ensure train and test directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

def clear_directory(directory):
    '''
    Clears sort directories to prevent duplicate or redundant data.
    '''
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            print(f"Clearing {file_path}")
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Get all subdirectories in BASE_DIR (excluding 'train' and 'test' if they exist)
CATEGORIES = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d not in ['train', 'test']]

# Split ratio
TRAIN_RATIO = 0.7

if __name__ == "__main__":
    clear_directory(os.path.join(TRAIN_DIR)) # wipe dataset/train/*/*
    clear_directory(os.path.join(TEST_DIR))  # wipe dataset/test/*/*
    print("Filling sorted categories...")
    # Process each category
    for category in CATEGORIES:
        category_path = os.path.join(BASE_DIR, category)

        # Check if the category directory exists
        if not os.path.exists(category_path):
            print(f"Skipping {category_path}, directory not found.")
            continue

        # Create corresponding train and test subdirectories
        train_category_path = os.path.join(TRAIN_DIR, category)
        test_category_path = os.path.join(TEST_DIR, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # Get list of images in the category folder
        images = [f for f in os.listdir(category_path) \
                  if os.path.isfile(os.path.join(category_path, f))]

        # Shuffle images for pseudo-randomness
        random.shuffle(images)

        # Split into train and test
        split_index = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Copy images to their respective directories
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))

        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_category_path, img))

        print(f"Processed {category}: {len(train_images)} for training, {len(test_images)} for testing.")

    print("Dataset splitting complete.")
