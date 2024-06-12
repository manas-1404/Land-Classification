import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the saved model
model_path = r'C:\Users\lyq09mow\Code\best_model.keras'

# Paths to the test data
test_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\testing'
new_test_dir = r'C:\Users\lyq09mow\Code\madrid-es\30km\testing'  # New test dataset

# Initialize test data generator
def setup_test_data_generator(test_dir, batch_size=16, target_size=(224, 224)):
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return test_generator

# Load the best saved model
model = load_model(model_path)

# Setup test data generators
test_generator = setup_test_data_generator(test_dir, batch_size=16)
new_test_generator = setup_test_data_generator(new_test_dir, batch_size=16)

# Evaluate the model using the test data generator
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Evaluate the model using the new test data generator
new_test_loss, new_test_accuracy = model.evaluate(new_test_generator)
print(f"New Test Loss: {new_test_loss}")
print(f"New Test Accuracy: {new_test_accuracy}")
