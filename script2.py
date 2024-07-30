import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import setup_data_generators
from resnet import build_resnet_101

# Paths to training, validation, testing data
train_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\training'
val_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\testing'

# Best hyperparameter based on fine-tuning results
learning_rate = 0.001
batch_size = 16
epochs = 100  

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Path to the saved model
model_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\86cent7_initial30.keras'

checkpoint_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\86cent7_middle120.keras'


# Load previous best model if exists
try:
    model = tf.keras.models.load_model(model_path)
    print("Loaded model from checkpoint.")
except:
    print("No checkpoint found, starting from scratch.")

for layer in model.layers:
    layer.trainable = False

# Total number of layers
total_layers = len(model.layers)
print("Total number of layers:", total_layers)

# Calculate the middle index
middle_index = total_layers // 2

# Calculate the starting and ending index for the middle 60 layers
start_index = middle_index - 60
end_index = middle_index + 60
print("Middle 120 layers range from index", start_index, "to", end_index)

# Unfreeze some layers for further training
for layer in model.layers[start_index:end_index]:
    layer.trainable = True

# Compile the model with the best hyperparameters
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001, verbose=1)
tqdm_callback = TqdmCallback(verbose=1)

# Determine the initial epoch
initial_epoch = 0
try:
    with open(r'C:\Users\lyq09mow\Model\Urban_Fabric\initial_epoch2.txt', 'r') as f:
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

# Save the current epoch
with open(r'C:\Users\lyq09mow\Model\Urban_Fabric\initial_epoch2.txt', 'w') as f:
    f.write(str(epochs - 1))

# Save the entire model for later use
model.save(f'C:\\Users\\lyq09mow\\Model\\Urban_Fabric\\middle120_{learning_rate}_bs{batch_size}_ep{epochs}.keras')

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

# Verify the class indices
print("Test Generator Class Indices:", test_generator.class_indices)

# Evaluate the model using the test data generator
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
