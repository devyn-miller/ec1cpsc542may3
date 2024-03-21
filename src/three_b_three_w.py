    #5
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
def three_b_three_w(model, test_dataset):
    """Evaluate and display the best and worst predictions of a model."""
    
    def display_images(images, titles, cols=3):
        """Display images with titles in a grid."""
        assert len(images) == len(titles)
        rows = len(images) // cols + int(len(images) % cols > 0)
        plt.figure(figsize=(cols * 4, rows * 4))
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
            plt.title(title)
            plt.axis('off')
        plt.show()

    def evaluate_and_display_best_worst_images(model, dataset, num_images=3):
        """Evaluate and display the best and worst predictions of a model."""
        accuracies = []
        images = []
        masks = []
        predictions = []

        # Iterate over the dataset to gather predictions and their true labels
        for image_batch, mask_batch in dataset:
            pred_mask_batch = model.predict(image_batch)
            for image, true_mask, pred_mask in zip(image_batch, mask_batch, pred_mask_batch):
                # Calculate accuracy for each prediction
                pred_mask_argmax = tf.math.argmax(pred_mask, axis=-1)
                accuracy = np.mean(true_mask.numpy().flatten() == pred_mask_argmax.numpy().flatten())
                accuracies.append(accuracy)
                images.append(image)
                masks.append(true_mask)
                predictions.append(pred_mask_argmax)

        # Sort predictions by accuracy
        sorted_indices = np.argsort(accuracies)
        best_indices = sorted_indices[-num_images:]
        worst_indices = sorted_indices[:num_images]

        # Display best predictions
        best_images = [images[i] for i in best_indices]
        best_titles = [f"Best {i+1}: Acc={accuracies[i]:.2f}" for i in best_indices]
        display_images(best_images, best_titles)

        # Display worst predictions
        worst_images = [images[i] for i in worst_indices]
        worst_titles = [f"Worst {i+1}: Acc={accuracies[i]:.2f}" for i in worst_indices]
        display_images(worst_images, worst_titles)
    evaluate_and_display_best_worst_images(model, test_dataset, num_images=3)
