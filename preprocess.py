from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Setup the training data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,   # Normalize pixel values to [0,1]
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,   # Randomly zoom image 
    horizontal_flip=True  # Randomly flip images
)

# Setup the validation and test data generators
test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation and test sets

# Prepare flow from directory for training
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\manas\data\madrid-es\training',  # This is the source directory for training images
    target_size=(224, 224),  # All images will be resized to 224x224
    batch_size=32,
    class_mode='categorical')  # Since we use categorical_crossentropy loss, we need categorical labels

# Prepare flow from directory for validation
validation_generator = test_datagen.flow_from_directory(
    r'C:\Users\manas\data\madrid-es\evalution',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

print("Preprocessing of images complete!")
