import os
import shutil

# Define the base directory where 'bare', 'blue', and 'white' folders are located
base_dir = '/home/an258456/Real-time-GesRec/datasets/HOLO/Video_data/subject3'

# List all subdirectories in the base directory
for subdir in os.listdir(base_dir):
    # Check if it's one of the directories of interest
    if subdir in ['bare', 'blue', 'white']:
        # Construct full path to this subdirectory
        subdir_path = os.path.join(base_dir, subdir)

        # Iterate over all class directories inside this subdirectory (class1 through class25)
        for class_dir in os.listdir(subdir_path):
            class_dir_path = os.path.join(subdir_path, class_dir)

            # Iterate over all r directories inside each class directory (r0 through r4)
            for r_dir in os.listdir(class_dir_path):
                r_dir_path = os.path.join(class_dir_path, r_dir)

                # Create sk_color_all folder inside each r directory
                sk_color_all_folder = os.path.join(r_dir_path, "sk_color_all")
                if not os.path.exists(sk_color_all_folder):
                    os.makedirs(sk_color_all_folder)

                # Move all images from current r directory to sk_color_all folder
                for file_name in os.listdir(r_dir_path):
                    if file_name.endswith('.jpg'):  # Assuming images are .jpg files. Change as needed.
                        shutil.move(os.path.join(r_dir_path, file_name), sk_color_all_folder)
