import tensorflow as tf
import matplotlib.pyplot as plt
from src.preprocessing import preprocess

def augment():  
    # Load preprocessed data and configuration variables.
    train_images, test_images, info, BATCH_SIZE, BUFFER_SIZE, STEPS_PER_EPOCH = preprocess()

    # Define a custom augmentation layer that inherits from tf.keras.layers.Layer.
    class Augment(tf.keras.layers.Layer):
        def __init__(self, seed=42):
            super().__init__()
            # Initialize random horizontal flip augmentation for both inputs and labels.
            # Using the same seed ensures identical transformations for inputs and labels.
            self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
            self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

        def call(self, inputs, labels):
            # Apply the augmentation to inputs and labels.
            inputs = self.augment_inputs(inputs)
            labels = self.augment_labels(labels)
            return inputs, labels

    # Prepare the training dataset with caching, shuffling, batching, and prefetching.
    # Apply the defined Augment layer to each batch of the dataset.
    train_batches = (
        train_images
        .cache()  # Cache the dataset in memory to improve performance.
        .shuffle(BUFFER_SIZE)  # Shuffle the dataset with a buffer size.
        .batch(BATCH_SIZE)  # Batch the dataset into specified batch sizes.
        .map(Augment())  # Apply the augmentation defined in the Augment class.
        .prefetch(buffer_size=tf.data.AUTOTUNE)  # Prefetch batches for performance.
    )

    # Prepare the test dataset by simply batching it. No augmentation is applied.
    test_batches = test_images.batch(BATCH_SIZE)

    # Define a function to visualize an image and its corresponding mask.
    def display(display_list):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))  # Convert tensor to image.
            plt.axis('off')  # Hide axis for clarity.
            plt.show()

    # Display sample images and masks from the augmented training batches.
    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]  # Take the first example from the batch.
        display([sample_image, sample_mask])  # Visualize the sample image and mask.

    # Return the training and testing batches, display function, and a sample image and mask for further use.
    return train_batches, test_batches, display, sample_image, sample_mask
