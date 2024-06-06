import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tqdm.keras import TqdmCallback

from preprocess import setup_data_generators
from resnet import build_resnet_101

# Assuming the 'setup_data_generators' function and 'build_resnet_50' are correctly defined/imported.

# Paths to training and validation data
train_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\training'
val_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\evaluation'
test_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\testing'

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir)

# Build and compile the model
# model = build_resnet_101(input_shape=(224, 224, 3), num_outputs=18)

#Load previous best model
model = tf.keras.models.load_model('best_model.keras')

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
tqdm_callback = TqdmCallback(verbose=1)  # TQDM callback for more interactive progress bar

# Train the model using the tqdm progress bar through the callback
history = model.fit(train_generator,
                    epochs=50,
                    validation_data=validation_generator,
                    callbacks=[checkpoint, early_stop, tqdm_callback],
                    verbose=0)  # Set verbose=0 to prevent default progress bar from showing

# Save the entire model for later use
model.save('madrid_land_model.keras')

print("Training Complete")


print("Testing the model now!")

model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model using the test data generator
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")