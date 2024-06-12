import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import setup_data_generators
from resnet import build_resnet_101

# Paths to training, validation, testing data
train_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\training'
val_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\evaluation'
test_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\testing'

new_test_dir = r'C:\Users\lyq09mow\Code\madrid-es\30km\testing'  # New test dataset

# Path to the saved model
# model_path = r'C:\Users\lyq09mow\Code\madrid_land_model.keras'
model_path = r'C:\Users\lyq09mow\Code\best_model.keras'

# Best hyperparameter based on fine-tuning results
learning_rate = 0.001
batch_size = 16
epochs = 100

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Load previous best model
model = tf.keras.models.load_model(model_path)

# Unfreeze more layers for fine-tuning
for layer in model.layers[-50:]:
    layer.trainable = True

# Compile the model with the best hyperparameters
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(r'C:\Users\lyq09mow\Code\madrid_land_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
tqdm_callback = TqdmCallback(verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop, reduce_lr, tqdm_callback],
    verbose=1
)

# Save the entire model for later use
model.save(f'C:\\Users\\lyq09mow\\Code\\madrid_land_model_lr{learning_rate}_bs{batch_size}_ep{epochs}.keras')

# Setup test data generators
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

test_generator = setup_test_data_generator(test_dir, batch_size=16)
new_test_generator = setup_test_data_generator(new_test_dir, batch_size=16)

# Verify the class indices
print("Test Generator Class Indices:", test_generator.class_indices)
print("New Test Generator Class Indices:", new_test_generator.class_indices)

# Evaluate the model using the test data generator
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Evaluate the model using the new test data generator
new_test_loss, new_test_accuracy = model.evaluate(new_test_generator)
print(f"New Test Loss: {new_test_loss}")
print(f"New Test Accuracy: {new_test_accuracy}")