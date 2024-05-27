import os
from tqdm import tqdm
import random
import shutil

class Sample:
    def __init__(self, refracted, reference, depth):
        self.refracted = refracted
        self.reference = reference
        self.depth_file = depth

class Filepaths:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.refracted_dir = os.path.join(root_dir, 'refraction')
        self.reference_dir = os.path.join(root_dir, 'reference')
        self.depth_files, self.refracted_files = self.get_file_paths()

    def get_file_id(self, file_path):
        """
        Extract the unique ID from the file name.
        """
        file_name = os.path.basename(file_path)
        if file_name.endswith('.npy'):  # Return the filename without the extension and split at _ to get the ID
            file_name = file_name.split('_')[1]
            return file_name.split('.')[0]
        elif file_name.endswith('.png'):  # Return the filename without the extension
            return file_name.split('.')[0]

    def get_reference_file(self, file_path, reference_dir):
        """
        Extract the reference pattern from the file name.
        """
        # Split the path by the '/' delimiter
        path_parts = file_path.split('/')
        # Assuming 'gray' is always in this position (third from the end)
        color = path_parts[-2]  # '-2' accesses the second-to-last element
        color_path = os.path.join(reference_dir, color + ".png")
        return color_path , color

    def get_file_paths(self, refracted_dir=None, depth_dir=None):
        refracted_dir = refracted_dir or self.refracted_dir
        depth_dir = depth_dir or self.depth_dir

        depth_files = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')]
        refracted_files = [os.path.join(root, f) for root, _, files in os.walk(refracted_dir) for f in files if f.endswith('.png')]

        print("Number of files in depth directory: ", len(depth_files))
        print("Number of files in refracted directory: ", len(refracted_files))

        return depth_files, refracted_files

    def match_samples(self, depth_files=None, refracted_files=None, reference_dir=None):
        """
        Create a TensorFlow dataset that loads refracted and reference images, depth maps,
        ensuring depth maps are paired with their corresponding refracted files based on the file ID.
        """
        depth_files = depth_files or self.depth_files
        refracted_files = refracted_files or self.refracted_files
        reference_dir = reference_dir or self.reference_dir

        matched_samples = []
        progress_bar = tqdm(total=len(depth_files), desc="Matching files", unit="file")

        allowed_colors = ['checkers', 'gray', 'gray2', 'rocky']
        for depth_file in depth_files:
            depth_id = self.get_file_id(depth_file)
            for refracted_file in refracted_files:
                refracted_id = self.get_file_id(refracted_file)
                if len(refracted_id) < 6:
                    refracted_id = '0'*(6-len(refracted_id)) + refracted_id

                if depth_id == refracted_id:
                    reference_file , color = self.get_reference_file(refracted_file, reference_dir)
                    if color in allowed_colors:
                        sample = Sample(refracted_file, reference_file, depth_file)
                        matched_samples.append(sample)
            progress_bar.update(1)
        progress_bar.close()

        print("Number of matched samples: ", len(matched_samples))
        return matched_samples

    def move_samples(samples, validation_dir):
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)

        for sample in samples:
            # Create subdirectories if they do not exist
            depth_dir = os.path.join(validation_dir, 'depth')
            refracted_subdir = os.path.basename(os.path.dirname(sample.refracted))
            refracted_dir = os.path.join(validation_dir, 'refraction', refracted_subdir)
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)
            if not os.path.exists(refracted_dir):
                os.makedirs(refracted_dir)

            # Define destination paths
            dest_path_depth = os.path.join(depth_dir, os.path.basename(sample.depth_file))
            dest_path_refracted = os.path.join(refracted_dir, os.path.basename(sample.refracted))

            # Move files
            if not os.path.exists(dest_path_depth):
                shutil.move(sample.depth_file, dest_path_depth)
                print(f"Moved depth file {sample.depth_file} to {dest_path_depth}")
            shutil.move(sample.refracted, dest_path_refracted)
            print(f"Moved refracted file {sample.refracted} to {dest_path_refracted}")

    def debug(root_dir):
        filepaths = Filepaths(root_dir)
        matched_samples = filepaths.match_samples(filepaths.depth_files, filepaths.refracted_files, filepaths.reference_dir)
        paired_depth_ids = []
        paired_refracted_ids = []

        for sample in matched_samples:
            depth_path = sample.depth_file
            refracted_path = sample.refracted

            paired_depth_id = filepaths.get_file_id(depth_path)
            paired_refracted_id = filepaths.get_file_id(refracted_path)

            paired_depth_ids.append(paired_depth_id)
            paired_refracted_ids.append(paired_refracted_id)

        unpaired_depth_ids = []
        for file in filepaths.depth_files:
            depth_id = filepaths.get_file_id(file)
            if depth_id not in paired_depth_ids:
                unpaired_depth_ids.append(depth_id)
        unpaired_depth_ids = sorted(unpaired_depth_ids)

        unpaired_refracted_ids = []
        for file in filepaths.refracted_files:
            refracted_id = filepaths.get_file_id(file)
            if refracted_id not in paired_refracted_ids:
                unpaired_refracted_ids.append(refracted_id)
        unpaired_refracted_ids = sorted(unpaired_refracted_ids)

        concatenated_ids = []
        for i in range(len(paired_depth_ids)):
            concatenated = [unpaired_depth_ids[i], unpaired_refracted_ids[i]]
            concatenated_ids.append(concatenated)

        for entry in concatenated_ids:
            print(entry)

if __name__ == "__main__":
    root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic/validation'
    Filepaths.debug(root_dir)

    # root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/homemade'
    # validation_dir = os.path.join(root_dir, 'validation')
    # filepaths = Filepaths(root_dir)
    # matched_samples = filepaths.match_samples(filepaths.depth_files, filepaths.refracted_files, filepaths.reference_dir)

    # # Randomly select 20% of all matched samples
    # num_validation_samples = int(0.2 * len(matched_samples))
    # validation_samples = random.sample(matched_samples, num_validation_samples)
    
    # # Move the selected samples to the validation directory
    # Filepaths.move_samples(validation_samples, validation_dir)
