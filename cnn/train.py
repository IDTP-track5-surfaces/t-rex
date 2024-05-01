import matplotlib.pyplot as plt
import tensorflow as tf

from model import FluidNet, create_model, depth_loss

from dataloader import load_and_preprocess_data
import numpy as np
from model import threshold_accuracy



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
    
def calculate_metrics(y_true, y_pred):
    # Calculate the accuracy metrics
    acc_125 = threshold_accuracy(y_true, y_pred, 1.25).numpy()
    acc_15625 = threshold_accuracy(y_true, y_pred, 1.25**2).numpy()
    acc_1953125 = threshold_accuracy(y_true, y_pred, 1.25**3).numpy()
    rmse = tf.keras.metrics.RootMeanSquaredError()(y_true, y_pred).numpy()
    mae = tf.keras.metrics.MeanAbsoluteError()(y_true, y_pred).numpy()

    return acc_125, acc_15625, acc_1953125, rmse, mae


if __name__ == "__main__":

    with tf.device('/gpu:0'):  # Use '/gpu:0' if TensorFlow-Metal is installed
        model , custom_objects = create_model()
        input_tensors, depth_tensors, normal_tensors = load_and_preprocess_data()
        history = train_model(model, input_tensors, depth_tensors)
        plot_metrics(history)
        model.save('fluid_net.h5')

        # Expand the depth tensors to have an extra dimension
        depth_tensors = np.expand_dims(depth_tensors, axis=-1)

        # Pick 3 random indices from the dataset
        random_indices = np.random.choice(input_tensors.shape[0], 3, replace=False)
        
        # Use tf.gather to select random indices for tensors
        test_input = tf.gather(input_tensors, random_indices)
        test_depth = tf.gather(depth_tensors, random_indices)

        # Inference test
        loaded_model = tf.keras.models.load_model('fluid_net.h5', custom_objects=custom_objects)
        test_predictions = loaded_model.predict(test_input)

        # Use tf.gather for refracted and reference tensors, assuming they are part of input_tensors and follow channels
        refracted_tensors = tf.gather(input_tensors[:, :, :, :3], random_indices)
        reference_tensors = tf.gather(input_tensors[:, :, :, 3:], random_indices)

        # Plot the results with the ground truth
        fig, axes = plt.subplots(3, 4, figsize=(12, 12))
        for i in range(3):
            acc_metrics = calculate_metrics(test_depth[i], test_predictions[i])
            acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics
            metric_str = f"δ1: {acc_125:.3f}, δ2: {acc_15625:.3f}, δ3: {acc_1953125:.3f}\nRMSE: {rmse:.3f}, ARE: {are:.3f}"
            
            axes[i, 0].imshow(refracted_tensors[i].numpy())
            axes[i, 0].set_title('Refracted Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(reference_tensors[i].numpy())
            axes[i, 1].set_title('Reference Image')
            axes[i, 1].axis('off')

            im_pred = axes[i, 2].imshow(test_predictions[i, :, :, 0], cmap='viridis')
            axes[i, 2].set_title('Predicted Depth')
            axes[i, 2].axis('off')

            im_gt = axes[i, 3].imshow(test_depth[i, :, :, 0], cmap='viridis')
            axes[i, 3].set_title('Ground Truth Depth')
            axes[i, 3].axis('off')

            plt.colorbar(im_pred, ax=axes[i, 2])
            plt.colorbar(im_gt, ax=axes[i, 3])
            
            axes[i, 2].annotate(metric_str, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                                 va='top', fontsize=8, color='white', backgroundcolor='black')

        plt.tight_layout()
        plt.show()
    


