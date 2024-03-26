import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class Sample:
    def __init__(self, refracted, reference, depth, normal):
        self.refracted = refracted
        self.reference = reference
        self.depth_file = depth
        self.normal_file = normal


def get_file_id(file_name):
    """
    Extract the unique ID from the file name.
    """
    return file_name.split('_')[1]

def get_reference_file(file_name):
    """
    Extract the reference pattern from the file name.
    """

    # Split the path by the '/' delimiter
    path_parts = file_name.split('/')

    # Assuming 'gray' is always in this position (third from the end)
    color = path_parts[-2]  # '-2' accesses the second-to-last element

    color_path = os.path.join("../data/validation/reference", color)
    return color_path


def get_file_paths(refracted_dir, depth_dir, normal_dir):
    # Create a dictionary to match depth files with normal files based on IDs
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith('.npy')]
    refracted_files = [os.path.join(root, file) for root, _, files in os.walk(refracted_dir) for file in files if file.endswith('.png')]

    print(len(depth_files))
    print(len(normal_files))
    print(len(refracted_files))

    return depth_files, normal_files, refracted_files

def load_and_preprocess_data(refracted_files, depth_files, normal_files):
    """
    Create a TensorFlow dataset that loads refracted and reference images, depth maps, and normal maps,
    ensuring depth maps are paired with their corresponding normal maps based on the file ID.
    """

    # Match depth, normal, and refracted files based on the file ID 
    matched_samples = []
    progress_bar = tqdm(total=len(depth_files), desc="Processing files", unit="file")
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
                        reference_file = get_reference_file(refracted_file)
                        sample = Sample(refracted_file, reference_file, depth_file, normal_file)
                        matched_samples.append(sample)


        progress_bar.update(1)
    progress_bar.close()

    return matched_samples

def debug():

    print("DEBUG_LOG: Starting debug function to check for unpaired files. Testing load_and_preprocess_data() function.")

    refracted_dir = "../data/validation/render"
    depth_dir = "../data/validation/depth_map"
    normal_dir = "../data/validation/normal_map"

    depth_files, normal_files, refracted_files = get_file_paths(refracted_dir, depth_dir, normal_dir)
    matched_samples = load_and_preprocess_data(refracted_files, depth_files, normal_files)
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
            
    refracted_dir = "../data/validation/render"
    depth_dir = "../data/validation/depth_map"
    normal_dir = "../data/validation/normal_map"

    depth_files, normal_files, refracted_files = get_file_paths(refracted_dir, depth_dir, normal_dir)
    matched_samples = load_and_preprocess_data(refracted_files, depth_files, normal_files)
    print("Number of matched samples: ", len(matched_samples))
        


    

        
    
