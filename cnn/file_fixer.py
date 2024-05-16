import os


def fix_refracted_dynamic():
    # Define the directory containing the files
    directory = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic/refraction"

    # Iterate over each file in the directory
    # directory has sub directories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.startswith("rgb_") and filename.endswith(".png"):
                # remove anything after the last "_"
                id = filename.split("_")[1]
                # Construct the full old and new file paths
                old_file = os.path.join(root, filename)
                new_file = os.path.join(root, id + ".png")
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")

def fix_normal_dynamic():
    # Define the directory containing the files
    directory = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic/normal"

    # Iterate over each file in the directory
    for file in os.listdir(directory):
        if file.startswith("normal_") and file.endswith(".npy"):
            # remove anything after the last "_"
            id = file.split("_")[1]
            # Construct the full old and new file paths
            old_file = os.path.join(directory, file)
            new_file = os.path.join(directory, "normal_" + id + ".npy")
            os.rename(old_file, new_file)
            print(f"Renamed '{old_file}' to '{new_file}'")

def fix_depth_dynamic():
    # Define the directory containing the files
    directory = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic/depth"

    # Iterate over each file in the directory
    for file in os.listdir(directory):
        if file.startswith("depth_") and file.endswith(".npy"):
            # remove anything after the last "_"
            id = file.split("_")[1]
            # Construct the full old and new file paths
            old_file = os.path.join(directory, file)
            new_file = os.path.join(directory, "depth_" + id + ".npy")
            if file.endswith(".npy.npy"):
                print("NOT GOOD")
                break
            os.rename(old_file, new_file)
            print(f"Renamed '{old_file}' to '{new_file}'")

# remove the extra .npy from the files
def remove_npy():
    # Define the directory containing the files
    directory = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/data/dynamic/depth"

    # Iterate over each file in the directory
    for file in os.listdir(directory):
        if file.endswith(".npy.npy"):
            # remove anything after the last "_"
            # Construct the full old and new file paths
            old_file = os.path.join(directory, file)
            new_file = os.path.join(directory, file[:-4])
            os.rename(old_file, new_file)
            print(f"Renamed '{old_file}' to '{new_file}'")

if __name__ == "__main__":
    print("What do you want to fix?")