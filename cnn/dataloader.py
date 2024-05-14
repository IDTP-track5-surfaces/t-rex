import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from filepaths import Filepaths

REFRACTED_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/data_sets/refraction"
DEPTH_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/data_sets/depth"
REFERENCE_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/data_sets/references"
NORMAL_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/data_sets/normal"


def load_numpy_array(file_path):
    """
    Load a numpy array from a file.
    """
    return np.load(file_path)

def load_image_as_tensor(image_path, image_size=(128, 128)):
    """
    Load an image file as a TensorFlow tensor in grayscale and resize it.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3) # Adjust channels for grayscale
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def preprocess_data(matched_samples):
    """
    Extending the function to load and preprocess data.
    """

    # Your existing matching logic here...

    # Convert matched samples into TensorFlow dataset
    refracted_tensors = []
    reference_tensors = []
    depth_tensors = []
    normal_tensors = []

    progress_bar = tqdm(total=len(matched_samples), desc="Preprocessing data", unit="sample")
    for sample in matched_samples:
        refracted_tensor = load_image_as_tensor(sample.refracted)
        reference_tensor = load_image_as_tensor(sample.reference)
        depth_tensor = tf.convert_to_tensor(load_numpy_array(sample.depth_file), dtype=tf.float32)
        normal_tensor = tf.convert_to_tensor(load_numpy_array(sample.normal_file), dtype=tf.float32)
        # # Optionally, resize depth and normal maps to match input size
        depth_tensor = tf.image.resize(depth_tensor[None, :, :, None], (128, 128))[0, :, :, 0]
        normal_tensor = tf.image.resize(normal_tensor[None, :, :, :], (128, 128))[0, :, :, :]

        refracted_tensors.append(refracted_tensor)
        reference_tensors.append(reference_tensor)
        depth_tensors.append(depth_tensor)
        normal_tensors.append(normal_tensor)

        progress_bar.update(1)

    progress_bar.close()

    # Stack tensors to create a batch dimension
    refracted_tensors = tf.stack(refracted_tensors)
    reference_tensors = tf.stack(reference_tensors)
    depth_tensors = tf.stack(depth_tensors)
    normal_tensors = tf.stack(normal_tensors)


    max_depth = tf.reduce_max(depth_tensors) # Normalize depth maps by the maximum depth value
    print("Max depth: ", max_depth)
    depth_tensors = depth_tensors / max_depth

    print("Refracted tensors shape: ", refracted_tensors.shape)
    print("Reference tensors shape: ", reference_tensors.shape)

    # Combine refracted and reference tensors along the channel dimension to match the model's expected input
    input_tensors = tf.concat([refracted_tensors, reference_tensors], axis=-1)

    return input_tensors, depth_tensors, normal_tensors

def load_and_preprocess_data(filepaths):
    """
    Load and preprocess refracted and reference images, depth maps, and normal maps from directories.
    """
    print("Getting file paths...")
    matched_samples = filepaths.match_samples()
    print("Number of matched samples: ", len(matched_samples))

    input_tensors, depth_tensors, normal_tensors = preprocess_data(matched_samples)
    print("Input tensors shape: ", input_tensors.shape)
    print("Depth tensors shape: ", depth_tensors.shape)
    print("Normal tensors shape: ", normal_tensors.shape)

    return input_tensors, depth_tensors, normal_tensors

if __name__ == "__main__":
    root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic'
    filepaths = Filepaths(root_dir)
    # Load and preprocess data
    load_and_preprocess_data(filepaths)


        


    

        
    
