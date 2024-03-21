import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess():
    """Preprocess the dataset for training and testing."""
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


    def normalize(input_image, input_mask):
        # Normalize the input image and mask
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    def load_image(datapoint):
        # Load the image and mask
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(
            datapoint['segmentation_mask'],
            (128, 128),
            method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    # Get the length of the training set and set the batch size
    # Then set the buffer size and steps per epoch
    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    
    # Map the load_image function to the training and testing datasets
    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_images, test_images, info, BATCH_SIZE, BUFFER_SIZE, STEPS_PER_EPOCH
