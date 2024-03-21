import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
# import early_stopping from tensorflow.keras.callbacks
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import clear_output
import json



def train_model(model, train_batches, test_batches, show_predictions, info, BATCH_SIZE, STEPS_PER_EPOCH):
    with open('model_history.json', 'r') as file:
        model_history = json.load(file)
    
    # Display the predictions after each epoch
    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            clear_output(wait=True)
            show_predictions()
            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

    # Define the number of epochs and validation subsplits
    EPOCHS = 50
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

    # If the model history is empty, train the model
    if model_history is None:
        model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=[DisplayCallback(), early_stopping])

    # Determine the loss and validation loss
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # Plot the training and validation loss
    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    return model_history

