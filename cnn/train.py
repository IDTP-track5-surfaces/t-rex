import argparse
import tensorflow as tf
from model import FluidNet, create_model, depth_loss
from filepaths import Filepaths
from dataloader import load_and_preprocess_data
import numpy as np
from model import threshold_accuracy
from utils import plot_metrics, calculate_metrics, plot_inference

def train_model(model, train_input_tensors, train_depth_tensors, val_input_tensors, val_depth_tensors, epochs=10, batch_size=32):
    # Expand dimensions of depth tensor from [5400,128,128] to [5400,128,128,1] to match normal tensor
    print("Number of training samples: ", train_input_tensors.shape[0])
    print("Number of validation samples: ", val_input_tensors.shape[0])

    # Train the model with the combined ground truth tensor
    history = model.fit(
        train_input_tensors, train_depth_tensors, 
        validation_data=(val_input_tensors, val_depth_tensors),
        epochs=epochs, 
        batch_size=batch_size
    )
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate FluidNet model.")
    parser.add_argument('--root_dir', type=str, default='/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/', help='Root directory of the dataset.')
    parser.add_argument('--train_data', type=str, default='', help='Directory name for training data.')
    parser.add_argument('--augment', type=str, default='', help='Directory name for additional training data.')
    parser.add_argument('--val_data', type=str, default='', help='Directory name for validation data.')
    parser.add_argument('--load_model', type=str, default='', help='Path to load model for inference.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--infer', type=str, default=False, help='Infer on random samples from the validation dataset using the loaded model')

    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    TRAIN_DATA = args.train_data
    AUGMENT = args.augment
    LOAD_MODEL = args.load_model
    VAL_DATA = args.val_data
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print("starting...")
    with tf.device('/gpu:0'):  # Use '/gpu:0' if TensorFlow-Metal is installed
        print("Preparing validation data...")
        val_root_dir = ROOT_DIR + VAL_DATA + "/validation/" 
        val_filepaths = Filepaths(val_root_dir)
        val_input_tensors, val_output_tensors = load_and_preprocess_data(val_filepaths)

        model, custom_objects = create_model()
        if args.train_data != '':
            # Model training
            print("Preparing training data...")
            train_root_dir = ROOT_DIR + TRAIN_DATA + "/train/"
            train_filepaths = Filepaths(train_root_dir)
            train_input_tensors, train_output_tensors = load_and_preprocess_data(train_filepaths)


            if args.augment != '':
                print("Augmenting the data...")
                additional_train_root_dir = ROOT_DIR + AUGMENT + "/train/"
                additional_train_filepaths = Filepaths(additional_train_root_dir)
                additional_train_input_tensors, additional_train_output_tensors = load_and_preprocess_data(additional_train_filepaths)
                train_input_tensors = np.concatenate([train_input_tensors, additional_train_input_tensors], axis=0)
                train_output_tensors = np.concatenate([train_output_tensors, additional_train_output_tensors], axis=0)

            history = train_model(model, train_input_tensors, train_output_tensors, val_input_tensors, val_output_tensors, epochs=EPOCHS, batch_size=BATCH_SIZE)
            plot_metrics(history)
            model.save('fluid_net_big' + TRAIN_DATA + AUGMENT + '.h5')

        if args.infer:
            # Model Loading
            if LOAD_MODEL == '': 
                print("Please provide a path to the model to load for inference.")
            else:
                loaded_model = tf.keras.models.load_model(LOAD_MODEL, custom_objects=custom_objects)

                # Pick 3 random indices from the validation dataset
                random_indices = np.random.choice(val_input_tensors.shape[0], 3, replace=False)
                print("Random indices: ", random_indices)
                # Use tf.gather to select random indices for tensors
                infer_input = tf.gather(val_input_tensors, random_indices)
                infer_depth = tf.gather(val_output_tensors, random_indices)

                # Inference test
                infer_predictions = loaded_model.predict(infer_input)

                # Use tf.gather for refracted and reference tensors, assuming they are part of input_tensors and follow channels
                infer_refracted = tf.gather(val_input_tensors[:, :, :, :3], random_indices) # Adjust for grayscale images
                infer_reference = tf.gather(val_input_tensors[:, :, :, 3:], random_indices) # Adjust for grayscale images

                # Plot the results with the ground truth
                plot_inference(infer_refracted, infer_reference, infer_depth, infer_predictions)
