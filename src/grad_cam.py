    #8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def grad_cam(model, test_dataset):
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions. Ensure model.inputs is correctly passed.
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,  # Corrected this line
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            print(preds.shape)
            # if pred_index is None:
            #     pred_index = tf.argmax(preds[0])
            #     pred_index = pred_index.numpy()  # Convert to numpy if it's a tensor
            #     if isinstance(pred_index, np.ndarray):  # Ensure it's a scalar
            #         pred_index = pred_index.item()
            pred_index = 0
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with respect to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel in the feature map array by "how important this channel is"
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap between 0 & 1 for visualization
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def display_gradcam(img, heatmap, alpha=0.4):
    # Load the original image
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Display Grad CAM
        plt.imshow(superimposed_img)
        plt.show()

    def evaluate_and_display_best_worst_images_with_gradcam(model, dataset, last_conv_layer_name, num_images=3):
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
                images.append(image.numpy())
                masks.append(true_mask)
                predictions.append(pred_mask_argmax)

        # Sort predictions by accuracy
        sorted_indices = np.argsort(accuracies)
        best_indices = sorted_indices[-num_images:]
        worst_indices = sorted_indices[:num_images]

        # Prepare to display images and their Grad-CAM heatmaps
        selected_images = [images[i] for i in np.concatenate([best_indices, worst_indices])]
        selected_titles = [f"Best {i+1}" for i in range(num_images)] + [f"Worst {i+1}" for i in range(num_images)]

        cols = 3
        rows = (len(selected_images) // cols) + int(len(selected_images) % cols > 0)
        plt.figure(figsize=(cols * 4, rows * 4))

        for i, image in enumerate(selected_images):
            plt.subplot(rows, cols, i + 1)
            img_array = np.expand_dims(image, axis=0)
            # Use a fixed pred_index for visualization
            pred_index = 0
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)
            display_gradcam(image, heatmap)
            plt.title(selected_titles[i])

        plt.show()
        return selected_images, selected_titles

    # Assuming 'last_conv_layer_name' is the name of the last conv layer in your model
    last_conv_layer_name = 'conv2d_37'
    for image, mask in test_dataset.take(1):
        img_array = tf.expand_dims(image[0], axis=0)  # Select the first image in the batch
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        evaluate_and_display_best_worst_images_with_gradcam(model, test_dataset, last_conv_layer_name, num_images=3)

