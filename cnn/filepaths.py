import os
from tqdm import tqdm

class Sample:
    def __init__(self, refracted, reference, depth, normal):
        self.refracted = refracted
        self.reference = reference
        self.depth_file = depth
        self.normal_file = normal


class Filepaths:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.normal_dir = os.path.join(root_dir, 'normal')
        self.refracted_dir = os.path.join(root_dir, 'refraction')
        self.reference_dir = os.path.join(root_dir, 'reference')
        self.depth_files, self.normal_files, self.refracted_files = self.get_file_paths()


    def get_file_id(self, file_path):
        """
        Extract the unique ID from the file name.
        """
        file_name = os.path.basename(file_path)

        if file_name.endswith('.npy'): # return the filename without the extension and split at _ to get the ID
            file_name = file_name.split('_')[1]
            return file_name.split('.')[0]

        elif file_name.endswith('.png'): # return the filename without the extension
            return file_name.split('.')[0]
    

    def get_reference_file(self, file_path , reference_dir):
        """
        Extract the reference pattern from the file name.
        """

        # Split the path by the '/' delimiter
        path_parts = file_path.split('/')

        # Assuming 'gray' is always in this position (third from the end)
        color = path_parts[-2]  # '-2' accesses the second-to-last element

        color_path = os.path.join(reference_dir, color + ".png")
        return color_path


    def get_file_paths(self, refracted_dir=None, depth_dir=None, normal_dir=None):
        refracted_dir = refracted_dir or self.refracted_dir
        depth_dir = depth_dir or self.depth_dir
        normal_dir = normal_dir or self.normal_dir

        # Create a dictionary to match depth files with normal files based on IDs
        depth_files = [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.npy')]
        normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.npy')]
        refracted_files = [os.path.join(root, f) for root, _, files in os.walk(refracted_dir) for f in files if f.endswith('.png')]

        print("Number of files in depth directory: ", len(depth_files))
        print("Number of files in normal directory: ", len(normal_files))
        print("Number of files in refracted directory: ", len(refracted_files))

        return depth_files, normal_files, refracted_files

    def match_samples(self, depth_files = None, normal_files = None, refracted_files = None, reference_dir = None):
        """
        Create a TensorFlow dataset that loads refracted and reference images, depth maps, and normal maps,
        ensuring depth maps are paired with their corresponding normal maps based on the file ID.
        """
        
        depth_files = depth_files or self.depth_files
        normal_files = normal_files or self.normal_files
        refracted_files = refracted_files or self.refracted_files
        reference_dir = reference_dir or self.reference_dir


        # Match depth, normal, and refracted files based on the file ID 
        matched_samples = []
        progress_bar = tqdm(total=len(depth_files), desc="Matching files", unit="file")
        for depth_file in depth_files:
            depth_id = self.get_file_id(depth_file)
            # print("Depth ID: ", depth_id)
            for normal_file in normal_files:
                normal_id = self.get_file_id(normal_file)
                # print("Normal ID: ", normal_id)
                if depth_id == normal_id:
                    for refracted_file in refracted_files:
                        # grid_name, refracted_id = self.split_filename(refracted_file)
                        refracted_id = self.get_file_id(refracted_file)

                        if len(refracted_id) < 6:
                            refracted_id = '0'*(6-len(refracted_id)) + refracted_id

                        if depth_id == refracted_id:
                            # print("Matched refracted file")

                            reference_file = self.get_reference_file(refracted_file, reference_dir)
                            sample = Sample(refracted_file, reference_file, depth_file, normal_file)
                            matched_samples.append(sample)


            progress_bar.update(1)
        progress_bar.close()

        return matched_samples

    def split_filename(self,full_path):
        # Extract the filename from the full path
        filename = os.path.basename(full_path)
        filename, _ = os.path.splitext(filename)
        
        # Split the filename by underscore
        parts = filename.split('_')
        
        # The first part (grid number)
        grid_part = parts[0]
        
        # Join the remaining parts back with underscores
        remaining_part = '_'.join(parts[1:])
        # print("Grid part: ", grid_part)
        # print("Remaining part: ", remaining_part)
        return grid_part, remaining_part
    

if __name__ == "__main__":
    root_dir = '/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/homemade'
    filepaths = Filepaths(root_dir)
    matched_samples = filepaths.match_samples(filepaths.depth_files, filepaths.normal_files, filepaths.refracted_files, filepaths.reference_dir)

    # paired_depth_ida = []
    # paired_normal_ida = []
    # paired_refracted_ida = []

    # for sample in matched_samples:
    #     depth_path = sample.depth_file
    #     normal_path = sample.normal_file
    #     refracted_path = sample.refracted

    #     paired_depth_id = filepaths.get_file_id(depth_path)
    #     paired_normal_id = filepaths.get_file_id(normal_path)
    #     paired_refracted_id = filepaths.get_file_id(refracted_path)

    #     paired_depth_ida.append(paired_depth_id)
    #     paired_normal_ida.append(paired_normal_id)
    #     paired_refracted_ida.append(paired_refracted_id)

    # unpaired_depth_ida = []
    # for file in filepaths.depth_files:
    #     depth_id = filepaths.get_file_id(file)
    #     if depth_id not in paired_depth_ida:
    #         unpaired_depth_ida.append(depth_id)
    # unpaired_depth_ida = sorted(unpaired_depth_ida)

    # unpaired_normal_ida = []
    # for file in filepaths.normal_files:
    #     normal_id = filepaths.get_file_id(file)
    #     if normal_id not in paired_normal_ida:
    #         unpaired_normal_ida.append(normal_id)
    # unpaired_normal_ida = sorted(unpaired_normal_ida)

    # unpaired_refracted_ida = []
    # for file in filepaths.refracted_files:
    #     refracted_id = filepaths.get_file_id(file)
    #     if refracted_id not in paired_refracted_ida:
    #         unpaired_refracted_ida.append(refracted_id)
    # unpaired_refracted_ida = sorted(unpaired_refracted_ida)


    # concatenated_ida = []
    # for i in range(len(paired_depth_ida)):
    #     concatenated = [unpaired_depth_ida[i], unpaired_normal_ida[i], unpaired_refracted_ida[i]]
    #     concatenated_ida.append(concatenated)

    # for entry in concatenated_ida:
    #     print(entry)
    

    print("Number of matched samples: ", len(matched_samples))