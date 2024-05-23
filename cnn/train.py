
import tensorflow as tf

from model import FluidNet, create_model, depth_loss
from filepaths import Filepaths

from dataloader import load_and_preprocess_data
import numpy as np
from model import threshold_accuracy
from utils import plot_metrics, calculate_metrics
import matplotlib.pyplot as plt


def train_model(model, train_input_tensors, train_depth_tensors, val_input_tensors, val_depth_tensors, epochs=10, batch_size=32):
    # Expand dimensions of depth tensor from [5400,128,128] to [5400,128,128,1] to match normal tensor
    print("number of training samples: ", train_input_tensors.shape[0])
    print("number of validation samples: ", val_input_tensors.shape[0])

    # Train the model with the combined ground truth tensor
    history = model.fit(
        train_input_tensors, train_depth_tensors, 
        validation_data=(val_input_tensors, val_depth_tensors),
        epochs=epochs, 
        batch_size=batch_size
    )
    return history

ROOT_DIR = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/'

TRAIN_DATA = "dynamic"
# LOAD_MODEL = "dynamic"
VAL_DATA = "homemade"

if __name__ == "__main__":

    with tf.device('/gpu:0'):  # Use '/gpu:0' if TensorFlow-Metal is installed

        # # Model training
        train_root_dir = ROOT_DIR + TRAIN_DATA + "/train/"
        train_filepaths = Filepaths(train_root_dir)
        train_input_tensors, train_depth_tensors, train_normal_tensors = load_and_preprocess_data(train_filepaths)
        train_depth_tensors = np.expand_dims(train_depth_tensors, axis=-1)

        ## Augmenting the data.
        additional_train_root_dir = ROOT_DIR + "homemade/train/"
        additional_train_filepaths = Filepaths(additional_train_root_dir)
        additional_train_input_tensors, additional_train_depth_tensors, additional_train_normal_tensors = load_and_preprocess_data(additional_train_filepaths)
        additional_train_depth_tensors = np.expand_dims(additional_train_depth_tensors, axis=-1)
        train_input_tensors_augemented = np.concatenate((train_input_tensors, additional_train_input_tensors), axis=0)
        train_depth_tensors_augmented = np.concatenate((train_depth_tensors, additional_train_depth_tensors), axis=0)
        ####

        val_root_dir = ROOT_DIR + VAL_DATA + "/validation/"
        val_filepaths = Filepaths(val_root_dir)
        val_input_tensors, val_depth_tensors, val_normal_tensors = load_and_preprocess_data(val_filepaths)
        val_depth_tensors = np.expand_dims(val_depth_tensors, axis=-1)

        model , custom_objects = create_model()
        
        history = train_model(model, train_input_tensors_augemented, train_depth_tensors_augmented, val_input_tensors, val_depth_tensors, epochs=10, batch_size=32)
        plot_metrics(history)
        model.save('fluid_net_both' + '.h5')


        # Pick 3 random indices from the validation dataset
        random_indices = np.random.choice(val_input_tensors.shape[0], 3, replace=False)
        
        # # Use tf.gather to select random indices for tensors
        infer_input = tf.gather(val_input_tensors, random_indices)
        infer_depth = tf.gather(val_depth_tensors, random_indices)

        # # Inference test
        # loaded_model = tf.keras.models.load_model('fluid_net_'+ LOAD_MODEL + '.h5', custom_objects=custom_objects)
        loaded_model = model
        infer_predictions = loaded_model.predict(infer_input)

        # # Use tf.gather for refracted and reference tensors, assuming they are part of input_tensors and follow channels
        infer_refracted = tf.gather(val_input_tensors[:, :, :, :3], random_indices) # Adjust for grayscale images
        infer_reference = tf.gather(val_input_tensors[:, :, :, 3:], random_indices) # Adjust for grayscale images

        # Plot the results with the ground truth
        fig, axes = plt.subplots(3, 4, figsize=(12, 12))
        for i in range(3):
            acc_metrics = calculate_metrics(infer_depth[i], infer_predictions[i])
            acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics
            metric_str = f"δ1: {acc_125:.3f}, δ2: {acc_15625:.3f}, δ3: {acc_1953125:.3f}\nRMSE: {rmse:.3f}, ARE: {are:.3f}"
            
            axes[i, 0].imshow(infer_refracted[i].numpy())
            axes[i, 0].set_title('Refracted Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(infer_reference[i].numpy())
            axes[i, 1].set_title('Reference Image')
            axes[i, 1].axis('off')

            im_pred = axes[i, 2].imshow(infer_predictions[i, :, :, 0], cmap='viridis')
            axes[i, 2].set_title('Predicted Depth')
            axes[i, 2].axis('off')

            im_gt = axes[i, 3].imshow(infer_depth[i, :, :, 0], cmap='viridis')
            axes[i, 3].set_title('Ground Truth Depth')
            axes[i, 3].axis('off')

            plt.colorbar(im_pred, ax=axes[i, 2])
            plt.colorbar(im_gt, ax=axes[i, 3])
            
            axes[i, 2].annotate(metric_str, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                                 va='top', fontsize=8, color='white', backgroundcolor='black')

        plt.tight_layout()
        plt.show()



