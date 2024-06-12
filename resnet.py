import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Add
)
from tensorflow.keras.regularizers import l2

def _bn_relu(input):
    """Helper function to apply Batch Normalization followed by ReLU."""
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def _conv_bn_relu(filters, kernel_size, strides=(1, 1)):
    """Helper function to apply Convolution followed by Batch Normalization and ReLU."""
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=l2(1.e-4))(input)
        return _bn_relu(conv)
    return f

def _bn_relu_conv(filters, kernel_size, strides=(1, 1)):
    """Helper function to apply Batch Normalization, ReLU, and Convolution."""
    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=l2(1.e-4))(activation)
    return f

def _shortcut(input, residual):
    """Apply a shortcut connection between input and residual block and merge with Add."""
    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                          strides=(stride_width, stride_height), padding="valid",
                          kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(input)
    return Add()([shortcut, residual])

def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Build a residual block with repeating bottleneck layers."""
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters, init_strides, is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input
    return f

def basic_block(filters, strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic block for ResNet-18 and ResNet-34."""
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=strides,
                           padding="same", kernel_initializer="he_normal",
                           kernel_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv(filters, (3, 3), strides)(input)

        residual = _bn_relu_conv(filters, (3, 3))(conv1)
        return _shortcut(input, residual)
    return f

def bottleneck(filters, strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck block for ResNet-50, ResNet-101, and ResNet-152."""
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides,
                              padding="same", kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters, (1, 1), strides)(input)

        conv_3_3 = _bn_relu_conv(filters, (3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters * 4, (1, 1))(conv_3_3)
        return _shortcut(input, residual)
    return f

def ResNetBuilder(input_shape, num_outputs, block_fn, repetitions):
    """Build ResNet architecture based on the block function and repetitions specified."""
    img_input = Input(shape=input_shape)
    conv1 = _conv_bn_relu(64, (7, 7), strides=(2, 2))(img_input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    block = pool1
    filters = 64
    for i, r in enumerate(repetitions):
        block = _residual_block(block_fn, filters, r, is_first_layer=(i == 0))(block)
        filters *= 2

    block = _bn_relu(block)
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(flatten1)

    model = Model(inputs=img_input, outputs=dense)
    return model

def build_resnet_18(input_shape, num_outputs):
    """Build ResNet-18 architecture."""
    return ResNetBuilder(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

def build_resnet_34(input_shape, num_outputs):
    """Build ResNet-34 architecture."""
    return ResNetBuilder(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

def build_resnet_50(input_shape, num_outputs):
    """Build ResNet-50 architecture."""
    return ResNetBuilder(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

def build_resnet_101(input_shape, num_outputs):
    """Build ResNet-101 architecture."""
    return ResNetBuilder(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

def build_resnet_152(input_shape, num_outputs):
    """Build ResNet-152 architecture."""
    return ResNetBuilder(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

print("ResNet completed")