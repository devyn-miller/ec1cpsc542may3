import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

def model_compile():
    """## Define the model
    The model being used here is a modified [U-Net](https://arxiv.org/abs/1505.04597). A U-Net consists of an encoder (downsampler) and decoder (upsampler). To learn robust features and reduce the number of trainable parameters, use a pretrained model—[MobileNetV2](https://arxiv.org/abs/1801.04381)—as the encoder. For the decoder, you will use the upsample block, which is already implemented in the [pix2pix](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py) example in the TensorFlow Examples repo. (Check out the [pix2pix: Image-to-image translation with a conditional GAN](../generative/pix2pix.ipynb) tutorial in a notebook.)

    As mentioned, the encoder is a pretrained MobileNetV2 model. You will use the model from `tf.keras.applications`. The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.
    """

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    """The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples:"""

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    def unet_model(output_channels:int):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # Downsampling through the model
        x = inputs
        skips = []
        for filter_size in [64, 128, 256, 512]:
            x = tf.keras.layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            skips.append(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Dropout(0.5)(x)

        # Middle part of the network
        x = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        # Upsampling and establishing the skip connections
        skips = reversed(skips)
        for filter_size, skip in zip([512, 256, 128, 64], skips):
            x = tf.keras.layers.Conv2DTranspose(filter_size, 2, strides=2, padding='same')(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
            x = tf.keras.layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv2D(filter_size, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)

        # This is the last layer of the model
        last = tf.keras.layers.Conv2D(output_channels, 3, padding='same')  # Maintain 128x128 resolution

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    """Note that the number of filters on the last layer is set to the number of `output_channels`. This will be one output channel per class.

    ## Train the model

    Now, all that is left to do is to compile and train the model.

    Since this is a multiclass classification problem, use the `tf.keras.losses.SparseCategoricalCrossentropy` loss function with the `from_logits` argument set to `True`, since the labels are scalar integers instead of vectors of scores for each pixel of every class.

    When running inference, the label assigned to the pixel is the channel with the highest value. This is what the `create_mask` function is doing.
    """

    OUTPUT_CLASSES = 3
    model = tf.keras.models.load_model('/Users/devynmiller/the-final-assignment2-cpsc542/models/unet-model.h5')
    
    # model = unet_model(output_channels=OUTPUT_CLASSES)
    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])

    """Plot the resulting model architecture:"""

    tf.keras.utils.plot_model(model, show_shapes=True)
    return model
