import os

# Define the directory where your images are located
base_dir = '/home/an258456/Real-time-GesRec/datasets/HOLO/subject1/bare/class20/r2'

# Initialize the counter for image names
counter = 1

# Iterate over all subdirectories and files in your base directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Check if the file is a PNG image
        if file.endswith('.png'):
            # Get the full path of the source image and where to save it after renaming 
            full_file_path = os.path.join(root, file)
            new_file_name = f"{str(counter).zfill(5)}.png"
            new_file_path = os.path.join(root, new_file_name)
            
            # Rename the file
            os.rename(full_file_path, new_file_path)

            # Increment counter by 1 for next image name
            counter += 1

print("Image renaming is complete.")