import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the saved model
# model_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\85cent8.keras'

model_path = r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\85cent2_frankfurt.keras'

# Paths to the test data
# test_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\testing'

test_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom_frankfurt\testing'

new_test_dir1 = r'C:\Users\lyq09mow\ModelImages\berlin-de\25km_modification\testing' 

# new_test_dir1 = r'C:\Users\lyq09mow\ModelImages\frankfurt-de\testing' 
# new_test_dir2 = r'C:\Users\lyq09mow\ModelImages\barcelona-es\testing' 

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
new_test_generator1 = setup_test_data_generator(new_test_dir1, batch_size=16)
# new_test_generator2 = setup_test_data_generator(new_test_dir2, batch_size=16)

print("#"*50)
print(f"Model Name: {model_path}")

# Evaluate the model using the test data generator

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Dataset Name: {test_dir}")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

