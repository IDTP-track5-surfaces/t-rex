from __future__ import print_function



from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import *
import tensorflow as tf

import tensorflow.keras.backend as K

import os
import shutil
import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(history,date_time):
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
    plt.savefig("logs/" + str(date_time) + "/metrics.png")
    plt.show()
    
    return plt
    
def calculate_metrics(y_true, y_pred):
    # Calculate the accuracy metrics
    acc_125 = threshold_accuracy(y_true, y_pred, 1.25).numpy()
    acc_15625 = threshold_accuracy(y_true, y_pred, 1.25**2).numpy()
    acc_1953125 = threshold_accuracy(y_true, y_pred, 1.25**3).numpy()
    rmse = root_mean_squared_error(y_true, y_pred).numpy()
    are = absolute_relative_error(y_true, y_pred).numpy()

    return acc_125, acc_15625, acc_1953125, rmse, are 

def threshold_accuracy(y_true, y_pred, threshold=1.25):

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    ratio = tf.maximum(depth_true / depth_pred, depth_pred / depth_true)
    return tf.reduce_mean(tf.cast(ratio < threshold, tf.float32))

def root_mean_squared_error(y_true, y_pred): # RMSE on depth
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    # Calculate Root Mean Squared Error
    rmse = tf.sqrt(tf.reduce_mean(tf.square(depth_true - depth_pred)))
    return rmse

def absolute_relative_error(y_true, y_pred): # ARE on depth
    # Avoid division by zero
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    epsilon = tf.keras.backend.epsilon()  # Small constant for numerical stability
    # Calculate Absolute Relative Error
    are = tf.abs(depth_true - depth_pred) / (depth_true + epsilon)
    return tf.reduce_mean(are)

def plot_inference(infer_refracted, infer_reference, infer_true_output, infer_predictions, date_time):
    fig, axes = plt.subplots(3, 4, figsize=(12, 12))
    for i in range(3):
        true_output_expanded = tf.expand_dims(infer_true_output[i], axis=0)
        predictions_expanded = tf.expand_dims(infer_predictions[i], axis=0)
        acc_metrics = calculate_metrics(true_output_expanded, predictions_expanded)
        acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics
        metric_str = f"RMSE: {rmse:.3f} \n ARE: {are:.3f}"
        
        axes[i, 0].imshow(infer_refracted[i].numpy())
        axes[i, 0].set_title('Refracted Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(infer_reference[i].numpy())
        axes[i, 1].set_title('Reference Image')
        axes[i, 1].axis('off')

        im_pred = axes[i, 2].imshow(infer_predictions[i, :, :, 0], cmap='viridis')
        axes[i, 2].set_title('Predicted Depth')
        axes[i, 2].axis('off')

        im_gt = axes[i, 3].imshow(infer_true_output[i, :, :, 0], cmap='viridis')
        axes[i, 3].set_title('Ground Truth Depth')
        axes[i, 3].axis('off')

        plt.colorbar(im_pred, ax=axes[i, 2])
        plt.colorbar(im_gt, ax=axes[i, 3])
        
        axes[i, 2].annotate(metric_str, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                                va='top', fontsize=14, color='black', backgroundcolor='white')

    plt.tight_layout()
    plt.savefig("logs/" + str(date_time) + "/inference.png")
    plt.show()
    return plt

def save_history(history, filename):
    with open(filename, 'w') as f:
        # Header
        f.write('Epoch, Training Loss, Root Mean Squared Error, Absolute Relative Error, Accuracy 125, Accuracy 15625, Accuracy 1953125, Validation Loss, Validation Root Mean Squared Error, Validation Absolute Relative Error, Validation Accuracy 125, Validation Accuracy 15625, Validation Accuracy 1953125\n')
        
        # Data
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1}, "
                    f"{history.history['loss'][i]}, "
                    f"{history.history['root_mean_squared_error'][i]}, "
                    f"{history.history['absolute_relative_error'][i]}, "
                    f"{history.history['accuracy_125'][i]}, "
                    f"{history.history['accuracy_15625'][i]}, "
                    f"{history.history['accuracy_1953125'][i]}, "
                    f"{history.history['val_loss'][i]}, "
                    f"{history.history['val_root_mean_squared_error'][i]}, "
                    f"{history.history['val_absolute_relative_error'][i]}, "
                    f"{history.history['val_accuracy_125'][i]}, "
                    f"{history.history['val_accuracy_15625'][i]}, "
                    f"{history.history['val_accuracy_1953125'][i]}\n")

    

def plot_loss(history_file):
    # Load data from a text file
    data = pd.read_csv(history_file, header=0)
    # Remove any spaces from column names
    data.columns = data.columns.str.strip()

    # Extract data for plotting
    epochs = data['Epoch']
    training_loss = data['Training Loss']
    validation_loss = data['Validation Loss']

    # Create a plot with a logarithmic y-axis
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Show the plot
    plt.savefig("logs/loss.png")
    plt.show()


if __name__ == "__main__":
    plot_loss("/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/cnn/logs/20240603-181315/training_history.txt")
