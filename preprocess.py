from tensorflow.keras.preprocessing.image import ImageDataGenerator

def setup_data_generators(train_dir, val_dir, test_dir, batch_size=16, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=[1,1.2],
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=15,
        fill_mode='nearest'                
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        shuffle=True
    )

    test_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
        shuffle=False
    )
    
    print("\nPreprocessing of images complete!")
    
    return train_generator, validation_generator, test_generator