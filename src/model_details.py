import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tqdm.keras import TqdmCallback
from preprocess import setup_data_generators

# Paths to training, validation, testing data
train_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\training'
val_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\testing'

# Hyperparameters
learning_rate = 0.0001
batch_size = 16
epochs = 50

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Path to the saved pre-trained model
base_model_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\pretrained_resnet101.keras'

# Load the base model
base_model = tf.keras.models.load_model(base_model_path, compile=False)

# Add L2 regularization and dropout to the custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(12, activation='softmax', kernel_regularizer=l2(0.01))(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers in the base model first
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\pretrained_resnet101_regularzation_best_model.keras'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
tqdm_callback = TqdmCallback(verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop, reduce_lr, tqdm_callback],
    verbose=1
)

# Save the final model
model.save(f'C:\\Users\\lyq09mow\\Model\\Urban_Fabric\\pretrained_resnet101_regularzation_{learning_rate}_bs{batch_size}_ep{epochs}.keras')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
