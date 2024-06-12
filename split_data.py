import os
import shutil
from random import shuffle

def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15):
    """
    Split the data into training, validation, and test sets.
    Each set will have its own directory containing subdirectories for each class.

    Parameters:
    - source_dir: Directory containing the dataset's class folders.
    - train_dir: Directory where the training data subdirectories will be stored.
    - val_dir: Directory where the validation data subdirectories will be stored.
    - test_dir: Directory where the test data subdirectories will be stored.
    - train_size: Proportion of the training set (default 0.7).
    - val_size: Proportion of the validation set (default 0.15).
    """
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print("Classes found: ", classes)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        # Shuffle the images to ensure random distribution
        shuffle(images)

        # Calculate split indices
        train_end = int(len(images) * train_size)
        val_end = train_end + int(len(images) * val_size)

        # Split the images into groups
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Create class directories in train, val, and test
        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)

        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        # Function to copy images to the designated folders
        def copy_images(images, dest_dir):
            for img in images:
                src_path = os.path.join(cls_dir, img)
                dest_path = os.path.join(dest_dir, img)
                shutil.copy(src_path, dest_path)

        # Copy images to their respective folders
        copy_images(train_images, train_cls_dir)
        copy_images(val_images, val_cls_dir)
        copy_images(test_images, test_cls_dir)

# Usage
source_directory = r'C:\Users\lyq09mow\Data\madrid-es-images\30km'  # Path to the dataset directory
train_directory = r'C:\Users\lyq09mow\Code\madrid-es\30km\training'
val_directory = r'C:\Users\lyq09mow\Code\madrid-es\30km\evaluation'
test_directory = r'C:\Users\lyq09mow\Code\madrid-es\30km\testing'

split_data(source_directory, train_directory, val_directory, test_directory)
