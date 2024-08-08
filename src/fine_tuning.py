import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools

from preprocess import setup_data_generators
from resnet import build_resnet_101

# Assuming the 'setup_data_generators' function and 'build_resnet_50' are correctly defined/imported.

# Paths to training and validation data
train_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\training'         
val_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\testing'

# Path to the saved model
model_path = r'C:\Users\lyq09mow\Model\Urban_Fabric\pretrained_resnet101_barcelona_berlin_madrid_20unfreezed_best_model.keras'

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir)

# Define hyperparameter grid
learning_rates = [0.001, 0.0001, 0.00001]
batch_sizes = [16, 32]
epochs_list = [50, 100]

# Function to train and evaluate the model
def train_and_evaluate(learning_rate, batch_size, epochs):
    # Load previous best model
    model = tf.keras.models.load_model(model_path)

    # Compile the model with the current hyperparameters
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint(r'C:\Users\lyq09mow\Model\Urban_Fabric\fine_tuning_results\best_model_bar_ber_mad.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000001, verbose=1)
    tqdm_callback = TqdmCallback(verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr, tqdm_callback],
        verbose=0
    )

    # Save the entire model for later use
    model.save(f'C:\\Users\\lyq09mow\\Model\\Urban_Fabric\\Fine_Tuning\\madrid_land_model_lr{learning_rate}_bs{batch_size}_ep{epochs}.keras')

    # Evaluate the model using the test data generator
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    return test_loss, test_accuracy

# Iterate over hyperparameter combinations
results = []
for lr, bs, ep in itertools.product(learning_rates, batch_sizes, epochs_list):
    print(f"Training with lr={lr}, batch_size={bs}, epochs={ep}")
    loss, accuracy = train_and_evaluate(lr, bs, ep)
    results.append((lr, bs, ep, loss, accuracy))
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Print all results
for result in results:
    print(f"lr={result[0]}, batch_size={result[1]}, epochs={result[2]}, Test Loss={result[3]}, Test Accuracy={result[4]}")


