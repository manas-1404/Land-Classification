import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from tqdm import tqdm


def build_model(n_classes=17):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def setup_data_generators(train_dir, valid_dir, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')
    validation_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=batch_size, class_mode='categorical')
    return train_generator, validation_generator


def tqdm_generator(generator, desc, total):
    while True:
        with tqdm(total=total, desc=desc, unit="batch") as pbar:
            for x, y in generator:
                pbar.update(1)
                yield x, y


def train_model(model, train_generator, validation_generator, epochs=10):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_steps = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    model.fit(
        tqdm_generator(train_generator, desc="Training", total=train_steps),
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=tqdm_generator(validation_generator, desc="Validation", total=validation_steps),
        validation_steps=validation_steps)
    model.save("final_model.keras")
    return model


def fine_tune_model(model, train_generator, validation_generator):
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    train_steps = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size
    model.fit(
        tqdm_generator(train_generator, desc="Fine-tuning", total=train_steps),
        steps_per_epoch=train_steps,
        epochs=5,
        validation_data=tqdm_generator(validation_generator, desc="Validation", total=validation_steps),
        validation_steps=validation_steps)
    model.save("final_finetuned_model.keras")
    return model


if __name__ == "__main__":
    model = build_model(n_classes=17)
    train_gen, valid_gen = setup_data_generators(r'C:\Users\manas\data\madrid-es\training', r'C:\Users\manas\data\madrid-es\evalution')
    model = train_model(model, train_gen, valid_gen)
    model = fine_tune_model(model, train_gen, valid_gen)
