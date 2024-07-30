import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from preprocess import setup_data_generators
from resnet import build_resnet_101

# Paths to training, validation, testing data
train_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom_frankfurt\training'
val_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom_frankfurt\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom_frankfurt\testing'

# Best hyperparameter based on fine-tuning results
learning_rate = 0.001
batch_size = 16
epochs = 100  

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Path to the saved model
model_path = r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\85cent2.keras'
checkpoint_path = r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\85cent2_frankfurt.keras'

# Load previous best model if exists
try:
    model = tf.keras.models.load_model(model_path)
    print("Loaded model from checkpoint.")
except:
    model = build_resnet_101()  # Replace with your model initialization if not found
    print("No checkpoint found, starting from scratch.")

for layer in model.layers[:]:
    layer.trainable = False

# Unfreeze some layers for further training
for layer in model.layers[-20:]:
    layer.trainable = True

# Compile the model with the best hyperparameters
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000001, verbose=1)
tqdm_callback = TqdmCallback(verbose=1)

# Determine the initial epoch
initial_epoch = 0
try:
    with open(r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\initial_epoch3.txt', 'r') as f:
        initial_epoch = int(f.read().strip()) + 1
except:
    pass

# Train the model
history = model.fit(
    train_generator,
    initial_epoch=initial_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop, reduce_lr, tqdm_callback],
    verbose=1
)

# Function to plot the training history
def plot_training_history(history, save_dir):
    # Create a directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Save the plots
    plt.savefig(os.path.join(save_dir, 'training_performance.png'))
    plt.close()

# Plot and save the training history
plot_training_history(history, save_dir='C:\\Users\\lyq09mow\\Model\\Bar_Ber_Mad_Rom\\Performance')

# Save the current epoch
with open(r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\initial_epoch3.txt', 'w') as f:
    f.write(str(epochs - 1))

# Save the entire model for later use
model.save(f'C:\\Users\\lyq09mow\\Model\\Bar_Ber_Mad_Rom\\last25_{learning_rate}_bs{batch_size}_ep{epochs}.keras')

# Evaluate the model using the test data generator
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


with open(f'C:\\Users\\lyq09mow\\Model\\Bar_Ber_Mad_Rom\\Performance\\test_results.txt', 'w') as f:
    f.write(f'Test Loss: {test_loss}\n')
    f.write(f'Test Accuracy: {test_accuracy}\n')
