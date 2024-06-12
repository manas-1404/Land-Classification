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
train_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\training'
val_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\evaluation'
test_dir = r'C:\Users\lyq09mow\Code\madrid-es\50km\testing'

# Path to the saved model
model_path = r'C:\Users\lyq09mow\Code\best_model.keras'

# Initialize data generators
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir)

# Define hyperparameter grid
learning_rates = [0.001, 0.0001]
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
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
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
    model.save(f'C:\\Users\\lyq09mow\\Code\\madrid_land_model_lr{learning_rate}_bs{batch_size}_ep{epochs}.keras')

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



# Hyperparameter Fine Tuning results
# lr=0.001, batch_size=16, epochs=50, Test Loss=0.7245725989341736, Test Accuracy=0.7755259871482849
# lr=0.001, batch_size=16, epochs=100, Test Loss=0.7029247283935547, Test Accuracy=0.794634222984314
# lr=0.001, batch_size=32, epochs=50, Test Loss=0.8501923680305481, Test Accuracy=0.709708571434021
# lr=0.001, batch_size=32, epochs=100, Test Loss=0.8003213405609131, Test Accuracy=0.72881680727005
# lr=0.0001, batch_size=16, epochs=50, Test Loss=0.7025255560874939, Test Accuracy=0.7822813987731934
# lr=0.0001, batch_size=16, epochs=100, Test Loss=0.7107123732566833, Test Accuracy=0.7788071632385254
# lr=0.0001, batch_size=32, epochs=50, Test Loss=0.7166960835456848, Test Accuracy=0.7764910459518433
# lr=0.0001, batch_size=32, epochs=100, Test Loss=0.7175022959709167, Test Accuracy=0.7768770456314087