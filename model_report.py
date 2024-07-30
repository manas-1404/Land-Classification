import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import setup_data_generators

# Load the pre-trained model
model = load_model(r'C:\Users\lyq09mow\Model\Urban_Fabric\83cent4_best_model_120layer_bar_ber_mad.keras')

# Paths to test data
train_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\training'
val_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\evaluation'
test_dir = r'C:\Users\lyq09mow\ModelImages\Urban_Fabric\testing'

batch_size = 16

# Assuming setup_data_generators returns an ImageDataGenerator object
train_generator, validation_generator, test_generator = setup_data_generators(train_dir, val_dir, test_dir, batch_size=batch_size)

# Get the true labels and the predicted labels
true_labels = test_generator.classes
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
class_report = classification_report(true_labels, predicted_labels, target_names=test_generator.class_indices.keys())
print(class_report)
