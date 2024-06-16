import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from filepaths import Filepaths


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

def create_scale_data(depth_data, normal_data):
    """
    Create scale data for depth and normal maps.

    Args:
    - depth_data: Tensor of shape [batch_size, height, width, 1] - Depth map.
    - normal_data: Tensor of shape [batch_size, height, width, 3] - Normal map.

    Returns:
    - scale_data: Tensor of shape [batch_size, 1, 1, 4] - Contains min and max values for depth and normals.
    """
    # Calculate min and max for depth
    depth_min = tf.reduce_min(depth_data, axis=[1, 2], keepdims=True)
    depth_max = tf.reduce_max(depth_data, axis=[1, 2], keepdims=True)

    # Calculate min and max for normals
    normal_min = tf.reduce_min(normal_data, axis=[1, 2], keepdims=True)
    normal_max = tf.reduce_max(normal_data, axis=[1, 2], keepdims=True)

    # Concatenate min and max values for depth and normals
    scale_data = tf.concat([depth_min, depth_max, normal_min, normal_max], axis=-1)  # Shape: [batch_size, 1, 1, 4]

    return scale_data

def preprocess_data(matched_samples):
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

        depth_tensor = tf.image.resize(depth_tensor[None, :, :, None], (128, 128))[0]
        normal_tensor = tf.image.resize(normal_tensor[None, :, :, :], (128, 128))[0]

        refracted_tensors.append(refracted_tensor)
        reference_tensors.append(reference_tensor)
        depth_tensors.append(depth_tensor)
        normal_tensors.append(normal_tensor)

        progress_bar.update(1)
    progress_bar.close()

    refracted_tensors = tf.stack(refracted_tensors)
    reference_tensors = tf.stack(reference_tensors)
    depth_tensors = tf.stack(depth_tensors)
    normal_tensors = tf.stack(normal_tensors)

    # Debugging depth tensor values
    print("Depth tensor sample (post-resize):", depth_tensors[0, :5, :5, 0])

    max_depth_per_sample = tf.reduce_max(depth_tensors, axis=[1, 2], keepdims=True)
    normalized_depth_tensors = depth_tensors / max_depth_per_sample

    # Debug normalized depth tensor values
    print("Normalized Depth Tensors sample:", normalized_depth_tensors[0, :5, :5, 0])

    input_tensors = tf.concat([refracted_tensors, reference_tensors], axis=-1)


    output_tensors = tf.concat([normalized_depth_tensors, normal_tensors], axis=-1)

    return input_tensors, output_tensors



def load_and_preprocess_data(filepaths):
    """
    Load and preprocess refracted and reference images, depth maps, and normal maps from directories.
    """
    print("Getting file paths...")
    matched_samples = filepaths.match_samples()
    print("Number of matched samples: ", len(matched_samples))

    input_tensors, output_tensors = preprocess_data(matched_samples)
    print("Input tensors shape: ", input_tensors.shape)
    print("Output tensors shape: ", output_tensors.shape)

    return input_tensors, output_tensors

if __name__ == "__main__":
    root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/pool_homemade/train'
    filepaths = Filepaths(root_dir)
    # Load and preprocess data
    load_and_preprocess_data(filepaths)
