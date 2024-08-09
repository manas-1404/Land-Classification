import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Load your pre-trained model
model = load_model(r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\85cent2_frankfurt.keras')

# Set up the directories for the datasets
test_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom\testing'

# ImageDataGenerator for testing, no augmentation is used here
test_datagen = ImageDataGenerator(rescale=1./255)

# Create a data generator for the test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Function to generate Grad-CAM heatmaps
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Iterate over the images and find misclassifications
misclassified_imgs = []
for i in range(len(test_generator)):
    x, y_true = test_generator.next()
    y_pred = model.predict(x)
    predicted_class = np.argmax(y_pred[0], axis=-1)
    true_class = np.argmax(y_true[0], axis=-1)

    if predicted_class != true_class:
        # Image is misclassified
        misclassified_imgs.append((x, true_class, predicted_class))

# Select a misclassified image
img_array, true_class, pred_class = misclassified_imgs[0]  # For example, the first misclassified image

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, 'conv5_block3_out', pred_class)

# Display heatmap
plt.matshow(heatmap)
plt.show()

# Load the original image
img = np.squeeze(img_array)

# Superimpose the heatmap on original image
heatmap = np.uint8(255 * heatmap)
jet = plt.cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = np.clip(superimposed_img, 0, 1)

# Display the superimposed image
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()
