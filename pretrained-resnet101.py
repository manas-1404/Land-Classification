import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the pre-trained ResNet101 model
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# Unfreeze some layers for further training
# for layer in base_model.layers[-50:]:
#     layer.trainable = True

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the pre-trained model with custom layers
model.save(r'C:\Users\lyq09mow\Model\Bar_Ber_Mad_Rom\pretrained_resnet101.keras')

print("Pre-trained ResNet101 model saved as 'pretrained_resnet101.keras'")