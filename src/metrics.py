from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def evaluate_model(model, test_dataset, model_history):
    # Initial setup for manual loss calculation
    label = [0, 0]  # Dummy ground truth labels
    prediction = [[-3., 0], [-3, 0]]  # Dummy predictions (logits before softmax)
    sample_weight = [1, 10]  # Weights for each sample, emphasizing the second sample

    # Convert lists to tensors for TensorFlow processing
    label_tensor = tf.constant(label)
    prediction_tensor = tf.constant(prediction)
    sample_weight_tensor = tf.constant(sample_weight)

    # Define a loss function suitable for classification that takes logits as input
    # 'from_logits=True' indicates that the predictions are raw scores (logits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                        reduction=tf.keras.losses.Reduction.NONE)
    # Calculate the loss value manually using the defined loss function
    # This demonstrates how to apply sample weights in loss calculation
    loss_value = loss(label_tensor, prediction_tensor, sample_weight_tensor).numpy()
    print(loss_value)

    # Nested function to evaluate the model on the test dataset
    def evaluate_model(model, test_dataset):
        # Evaluate the model on the entire test dataset to get loss and accuracy
        loss, accuracy = model.evaluate(test_dataset)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

        # Initialize arrays to store true labels and predictions for metric calculation
        y_true = np.array([])
        y_pred = np.array([])
        # Iterate over the test dataset to collect predictions and true labels
        for datapoint in test_dataset.unbatch().batch(1).take(-1):
            image, mask = datapoint
            pred_mask = model.predict(image)
            # Flatten and append the true labels and predictions for each sample
            y_true = np.append(y_true, mask.numpy().flatten())
            y_pred = np.append(y_pred, np.argmax(pred_mask, axis=-1).flatten())

        # Calculate standard classification metrics using true labels and predictions
        conf_mat = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        # Initialize and update a MeanIoU metric object for Intersection over Union
        iou = tf.keras.metrics.MeanIoU(num_classes=3)
        iou.update_state(y_true, y_pred)
        iou_score = iou.result().numpy()

        # Print calculated metrics
        print(f'Confusion Matrix:\n{conf_mat}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'IoU: {iou_score}')

        # Visualize the confusion matrix and a bar chart of the metrics
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(conf_mat, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.bar(['Precision', 'Recall', 'F1', 'IoU'], [precision, recall, f1, iou_score])
        plt.title('Metrics')
        plt.show()

    # Call the nested function to evaluate the model
    evaluate_model(model, test_dataset)

    # Extract training history for accuracy and loss
    history_dict = model_history.history

    # Plot training and validation accuracy over epochs
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Plot training and validation loss over epochs
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()
