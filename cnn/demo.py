
import tensorflow as tf
from model import create_model
from filepaths import Filepaths
from dataloader import load_and_preprocess_data
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_metrics

def plot_inference(infer_refracted, infer_reference, infer_true_output, infer_predictions1, infer_predictions2, date_time= "Demo_181315-100437"):
    fig, axes = plt.subplots(3, 5, figsize=(12, 12))
    for i in range(3):
        true_output_expanded = tf.expand_dims(infer_true_output[i], axis=0)
        
        axes[i, 0].imshow(infer_refracted[i].numpy())
        axes[i, 0].set_title('Refracted Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(infer_reference[i].numpy())
        axes[i, 1].set_title('Reference Image')
        axes[i, 1].axis('off')

        im_pred_1 = axes[i, 2].imshow(infer_predictions1[i, :, :, 0], cmap='viridis')
        axes[i, 2].set_title('Predicted Depth \n TREX')
        axes[i, 2].axis('off')
        predictions_expanded1 = tf.expand_dims(infer_predictions1[i], axis=0)
        acc_metrics1 = calculate_metrics(true_output_expanded, predictions_expanded1)
        acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics1
        metric_str_1 = f"RMSE: {rmse:.3f}, \n ARE: {are:.3f}"
        axes[i, 2].annotate(metric_str_1, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                                va='top', fontsize=10, color='black', backgroundcolor='white')
        

        im_pred_2 = axes[i, 3].imshow(infer_predictions2[i, :, :, 0], cmap='viridis')
        axes[i, 3].set_title('Predicted Depth \n TREX-spline')
        axes[i, 3].axis('off')
        predictions_expanded2 = tf.expand_dims(infer_predictions2[i], axis=0)
        acc_metrics2 = calculate_metrics(true_output_expanded, predictions_expanded2)
        acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics2
        metric_str_2 = f"RMSE: {rmse:.3f}, \n ARE: {are:.3f}"
        axes[i, 3].annotate(metric_str_2, xy=(0.5, -0.1), xycoords='axes fraction', ha='center',
                                va='top', fontsize=10, color='black', backgroundcolor='white')

        im_gt = axes[i, 4].imshow(infer_true_output[i, :, :, 0], cmap='viridis')
        axes[i, 4].set_title('Ground Truth Depth')
        axes[i, 4].axis('off')

        plt.colorbar(im_pred_1, ax=axes[i, 2])
        plt.colorbar(im_pred_2, ax=axes[i, 3])
        plt.colorbar(im_gt, ax=axes[i, 4])

    plt.tight_layout()
    plt.savefig("logs/" + str(date_time) + "/inference.png")
    plt.show()
    return plt

if __name__ == "__main__":
    val_root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/pool_homemade/validation'
    LOAD_MODEL_1 = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/cnn/logs/20240603-100437/final_model.h5'
    LOAD_MODEL_2 = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/cnn/logs/20240603-181315/final_model.h5'


    val_filepaths = Filepaths(val_root_dir)
    val_input_tensors, val_output_tensors = load_and_preprocess_data(val_filepaths)
    model, custom_objects = create_model()


    # Pick 3 random indices from the validation dataset
    np.random.seed(3)
    random_indices = np.random.choice(val_input_tensors.shape[0], 3, replace=False)
    print("Random indices: ", random_indices)
    # Use tf.gather to select random indices for tensors
    infer_input = tf.gather(val_input_tensors, random_indices)
    infer_true_output = tf.gather(val_output_tensors, random_indices)
    print("Infer input shape: ", infer_input.shape)
    print("Infer true output shape: ", infer_true_output.shape)

    # Run inference on the validation dataset using the first model
    loaded_model_1 = tf.keras.models.load_model(LOAD_MODEL_1, custom_objects=custom_objects)
    # extract datetime of the saved model
   
    # Inference test
    infer_predictions_1 = loaded_model_1.predict(infer_input)
    print("Infer predictions shape: ", infer_predictions_1.shape)


    # Run inference on the validation dataset using the second model
    loaded_model_2 = tf.keras.models.load_model(LOAD_MODEL_2, custom_objects=custom_objects)

    # Inference test
    infer_predictions_2 = loaded_model_2.predict(infer_input)
    print("Infer predictions shape: ", infer_predictions_2.shape)

    # Use tf.gather for refracted and reference tensors, assuming they are part of input_tensors and follow channels
    infer_refracted = tf.gather(val_input_tensors[:, :, :, :3], random_indices) # Adjust for grayscale images
    infer_reference = tf.gather(val_input_tensors[:, :, :, 3:], random_indices) # Adjust for grayscale images
    plot_inference(infer_refracted, infer_reference, infer_true_output, infer_predictions_1, infer_predictions_2)



