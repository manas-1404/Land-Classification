import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from collections import Counter
from PIL import Image

def load_and_preprocess(img_path, new_shape=(224, 224), crop=None, channels="rgb", downsample=None):
    img = load_img(img_path, color_mode=channels)
    img = img_to_array(img)

    if new_shape:
        img = np.array(Image.fromarray(img.astype('uint8')).resize(new_shape))

    if crop:
        crop_x = (img.shape[0] - crop) // 2
        crop_y = (img.shape[1] - crop) // 2
        img = img[crop_x:crop_x + crop, crop_y:crop_y + crop]

    if downsample:
        img = img[::downsample, ::downsample, :]

    img = img.astype('float32') / 255.0  # Convert to float32 and rescale
    return img

def balanced_df(files, k=1, class_column=1):
    class_counts = Counter([f[class_column] for f in files])
    min_count = min(class_counts.values())
    balanced_files = []
    for class_index in class_counts:
        class_files = [f for f in files if f[class_column] == class_index]
        if len(class_files) > 0:
            class_files = np.array(class_files)
            selected_files = class_files[np.random.choice(len(class_files), min(min_count * k, len(class_files)), replace=False)]
            balanced_files.extend(selected_files)
    return balanced_files

def generator_from_directory(directory, image_generator=None, balance=None, class_mode='categorical',
                             batch_size=32, new_img_shape=(224, 224), crop=None, channels="rgb", shuffle=True):
    class_indices = {}
    for subdir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, subdir)):
            class_indices[subdir] = len(class_indices)
    
    all_files = []
    for class_name, class_index in class_indices.items():
        class_dir = os.path.join(directory, class_name)
        class_files = [(os.path.join(class_dir, fname), class_index) for fname in os.listdir(class_dir) if fname.lower().endswith(('jpg', 'jpeg', 'png'))]
        all_files.extend(class_files)

    if balance:
        all_files = balanced_df(all_files, k=balance)

    while True:
        if shuffle:
            np.random.shuffle(all_files)
        
        batch_files = all_files[:batch_size]
        all_files = all_files[batch_size:]
        if len(all_files) < batch_size:
            all_files = batch_files + all_files  # refill the list

        X = []
        y = []
        for file_path, class_index in batch_files:
            img = load_and_preprocess(file_path, new_shape=new_img_shape, crop=crop, channels=channels)
            X.append(img)
            y.append(class_index)
        
        X = np.array(X)
        y = np.array(y)
        
        if class_mode == 'categorical':
            y = to_categorical(y, num_classes=len(class_indices))
        
        if image_generator:
            for X_batch, y_batch in image_generator.flow(X, y, batch_size=batch_size, shuffle=False):
                break
        else:
            X_batch, y_batch = X, y
        
        yield X_batch, y_batch

def setup_data_generators(train_dir, val_dir, test_dir, batch_size=16, target_size=(224, 224)):
    train_augmenter = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=[1, 1.2],
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=15                    
    )

    test_augmenter = ImageDataGenerator()

    train_generator = generator_from_directory(train_dir, image_generator=train_augmenter, balance=1,
                                               new_img_shape=target_size, crop=None, batch_size=batch_size)

    validation_generator = generator_from_directory(val_dir, image_generator=test_augmenter, balance=1,
                                                    new_img_shape=target_size, crop=None, batch_size=batch_size)

    test_generator = generator_from_directory(test_dir, image_generator=test_augmenter, balance=1,
                                              new_img_shape=target_size, crop=None, batch_size=batch_size, shuffle=False)

    print("\nPreprocessing of images complete!")
    
    return train_generator, validation_generator, test_generator

# Example usage
# train_generator, validation_generator, test_generator = setup_data_generators('train', 'val', 'test')

# # Test data generators
# for X_batch, y_batch in train_generator:
#     print(f'Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}')
#     break  # Check only the first batch to ensure it's working
