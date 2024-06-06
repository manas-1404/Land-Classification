from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_data_generators(train_dir, val_dir, test_dir, batch_size=16, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=40,  # More aggressive augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.6, 1.4],
        channel_shift_range=0.2
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        workers=4,
        max_queue_size=10
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        workers=4,
        max_queue_size=10
    )

    test_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        workers=4,
        max_queue_size=10
    )
    
    print("\nPreprocessing of images complete!")
    
    return train_generator, validation_generator, test_generator