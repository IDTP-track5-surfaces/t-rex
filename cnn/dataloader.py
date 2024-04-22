import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

REFRACTED_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/validation/render"
DEPTH_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/validation/depth_map"
NORMAL_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/validation/normal_map"
REFERENCE_DIR = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/validation/reference"

class Sample:
    def __init__(self, refracted, reference, depth, normal):
        self.refracted = refracted
        self.reference = reference
        self.depth_file = depth
        self.normal_file = normal


def get_file_id(file_path):
    """
    Extract the unique ID from the file name.
    """
    file_name = os.path.basename(file_path)
    return file_name.split('_')[1]

def get_reference_file(file_path , reference_dir):
    """
    Extract the reference pattern from the file name.
    """

    # Split the path by the '/' delimiter
    path_parts = file_path.split('/')

    # Assuming 'gray' is always in this position (third from the end)
    color = path_parts[-2]  # '-2' accesses the second-to-last element

    color_path = os.path.join(reference_dir, color + ".png")
    return color_path


def get_file_paths(refracted_dir = REFRACTED_DIR, depth_dir = DEPTH_DIR, normal_dir = NORMAL_DIR):
    # Create a dictionary to match depth files with normal files based on IDs
    depth_files = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')]
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.npy')]
    refracted_files = [os.path.join(root, file) for root, _, files in os.walk(refracted_dir) for file in files if file.endswith('.png')]

    print("Number of files in depth directory: ", len(depth_files))
    print("Number of files in normal directory: ", len(normal_files))
    print("Number of files in refracted directory: ", len(refracted_files))

    return depth_files, normal_files, refracted_files

def load_numpy_array(file_path):
    """
    Load a numpy array from a file.
    """
    return np.load(file_path)

def load_image_as_tensor(image_path, image_size=(128, 128)):
    """
    Load an image file as a TensorFlow tensor and resize it.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_samples(refracted_files, depth_files, normal_files, reference_dir = REFERENCE_DIR):
    """
    Create a TensorFlow dataset that loads refracted and reference images, depth maps, and normal maps,
    ensuring depth maps are paired with their corresponding normal maps based on the file ID.
    """

    # Match depth, normal, and refracted files based on the file ID 
    matched_samples = []
    progress_bar = tqdm(total=len(depth_files), desc="Matching files", unit="file")
    for depth_file in depth_files:
        depth_id = get_file_id(depth_file)
        for normal_file in normal_files:
            normal_id = get_file_id(normal_file)
            if depth_id == normal_id:
                for refracted_file in refracted_files:
                    refracted_id = get_file_id(refracted_file)

                    if len(refracted_id) <6: # If the file id is less than 6 digits, append zeros to the front
                        refracted_id = '0'*(6-len(refracted_id)) + refracted_id

                    if depth_id == refracted_id:
                        reference_file = get_reference_file(refracted_file, reference_dir)
                        sample = Sample(refracted_file, reference_file, depth_file, normal_file)
                        matched_samples.append(sample)


        progress_bar.update(1)
    progress_bar.close()

    return matched_samples

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

    progress_bar = tqdm(total=len(matched_samples), desc="Loading and preprocessing data", unit="sample")
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

    print("Refracted tensors shape: ", refracted_tensors.shape)
    print("Reference tensors shape: ", reference_tensors.shape)

    # Combine refracted and reference tensors along the channel dimension to match the model's expected input
    input_tensors = tf.concat([refracted_tensors, reference_tensors], axis=-1)

    return input_tensors, depth_tensors, normal_tensors

def load_and_preprocess_data():
    """
    Load and preprocess refracted and reference images, depth maps, and normal maps from directories.
    """

    depth_files, normal_files, refracted_files = get_file_paths(REFRACTED_DIR, DEPTH_DIR, NORMAL_DIR)
    matched_samples = load_samples(refracted_files, depth_files, normal_files, REFERENCE_DIR)
    print("Number of matched samples: ", len(matched_samples))

    input_tensors, depth_tensors, normal_tensors = preprocess_data(matched_samples)
    print("Input tensors shape: ", input_tensors.shape)
    print("Depth tensors shape: ", depth_tensors.shape)
    print("Normal tensors shape: ", normal_tensors.shape)

    return input_tensors, depth_tensors, normal_tensors

def debug():

    print("DEBUG_LOG: Starting debug function to check for unpaired files. Testing load_and_preprocess_data() function.")

    depth_files, normal_files, refracted_files = get_file_paths(REFRACTED_DIR, DEPTH_DIR, NORMAL_DIR)
    matched_samples = load_samples(refracted_files, depth_files, normal_files, REFERENCE_DIR)
    print("Number of matched samples: ", len(matched_samples))



    paired_depth_ids = []
    paired_normal_ids = []
    paired_refracted_ids = []
    for sample in matched_samples:
        depth_path  = sample.depth_file
        normal_path = sample.normal_file
        refracted_path = sample.refracted

        paired_depth_id = get_file_id(depth_path)
        paired_normal_id = get_file_id(normal_path)
        paired_refracted_id = get_file_id(refracted_path)

        paired_depth_ids.append(paired_depth_id)
        paired_normal_ids.append(paired_normal_id)
        paired_refracted_ids.append(paired_refracted_id)
    
    unpaired_depth_id = []
    for file in depth_files:
        file_id = get_file_id(file)
        if file_id not in paired_depth_ids:
            unpaired_depth_id.append(file_id)
    unpaired_depth_ids = sorted(unpaired_depth_id)
    
    unpaired_normal_id = []
    for file in normal_files:
        file_id = get_file_id(file)
        if file_id not in paired_normal_ids:
            unpaired_normal_id.append(file_id)
    unpaired_normal_ids = sorted(unpaired_normal_id)

    unpaired_refracted_id = []
    for file in refracted_files:
        file_id = get_file_id(file)

        if not file_id.isdigit():
            print("File ID is not an integer: ", file_id , " in ", file)


        if file_id not in paired_refracted_ids:
            unpaired_refracted_id.append(file_id)
    unpaired_refracted_ids = sorted(unpaired_refracted_id)


    concatenatd_ids = []
    for i in range(len(unpaired_depth_ids)):
        concatenated = [unpaired_depth_ids[i], unpaired_normal_ids[i], unpaired_refracted_ids[i]]
        concatenatd_ids.append(concatenated)                    

    for entry in concatenatd_ids:
        print(entry)

    print("DEBUG_LOG: Number of unpaired files: ", len(concatenatd_ids))
    return 0

if __name__ == "__main__":

    # Load and preprocess data
    load_and_preprocess_data()
    # debug()


        


    

        
    
