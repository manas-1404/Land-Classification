from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def get_class_indices(train_dir, batch_size=16, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    return train_generator.class_indices

train_dir = r'C:\Users\lyq09mow\ModelImages\Bar_Ber_Mad_Rom\training'  # Update this path
class_indices = get_class_indices(train_dir)
print(class_indices)


# with open('class_indices.json', 'w') as f:
#     json.dump(class_indices, f)
