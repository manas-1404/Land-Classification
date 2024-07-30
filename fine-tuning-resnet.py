import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.utils import class_weight
import numpy as np
from preprocess import setup_data_generators
from tqdm import tqdm

# Paths to training, validation, testing data
train_dir = r'C:\Users\lyq09mow\ModelImages\Berlin_Madrid\training'
val_dir = r'C:\Users\lyq09mow\ModelImages\Berlin_Madrid\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Berlin_Madrid\testing'

batch_size = 16

# Set up the data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Calculate class weights
# true_labels = train_generator.classes
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(true_labels),
#     y=true_labels
# )
# class_weights_dict = dict(enumerate(class_weights))
# print("Class weights:", class_weights_dict)

# Load the pre-trained model
model = load_model(r'C:\Users\lyq09mow\Model\Berlin_Madrid\pretrained_resnet101.keras')



# Calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Define callbacks
checkpoint_path = r'C:\Users\lyq09mow\Model\Berlin_Madrid\pretrained_resnet101_berlin_madrid_best_model.keras'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
csv_logger = CSVLogger(r'C:\Users\lyq09mow\Model\Berlin_Madrid\training_log.csv', append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

# for layer in model.layers[:95]:
#     layer.trainable = False

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
epochs = 200
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=epochs,
          steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps,
          callbacks=[checkpoint, csv_logger, reduce_lr])

# Save the final model
final_model_path = r'C:\Users\lyq09mow\Model\Berlin_Madrid\berlin_madrid_resnet50_12class_25km_land_model.keras'
model.save(final_model_path)
print(f"Model saved as '{final_model_path}'")
