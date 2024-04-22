import matplotlib.pyplot as plt
import tensorflow as tf

from model import FluidNet, create_model
from utils import combined_loss
from dataloader import load_and_preprocess_data



def train_model(model, input_tensors, depth_tensors, epochs=10, batch_size=32):
    # Expand dimensions of depth tensor from [5400,128,128] to [5400,128,128,1] to match normal tensor
    depth_tensors_expanded = tf.expand_dims(depth_tensors, axis=-1)

    # Train the model with the combined ground truth tensor
    history = model.fit(input_tensors, depth_tensors_expanded,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2)
    
    return history


def plot_metrics(history):
    # Plot training & validation loss values
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot each of the other metrics
    metrics = [key for key in history.history.keys() if 'loss' not in key and 'val_' in key]
    for metric in metrics:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[metric.replace('val_', '')], label=f'Train {metric}')
        plt.plot(history.history[metric], label=f'Validation {metric}')
        plt.title('Model Metrics')
        plt.ylabel('Metric Value')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    



if __name__ == "__main__":

    with tf.device('/gpu:0'):  # Use '/gpu:0' if TensorFlow-Metal is installed
        model = create_model()
        input_tensors, depth_tensors, normal_tensors = load_and_preprocess_data()
        history = train_model(model, input_tensors, depth_tensors)
        plot_metrics(history)
        

    


