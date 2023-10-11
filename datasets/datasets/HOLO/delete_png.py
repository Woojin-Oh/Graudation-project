import os
from PIL import Image

# Define the directory where your images are located
base_dir = '/home/an258456/Real-time-GesRec/datasets/HOLO/subject1'

# Iterate over all subdirectories and files in your base directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Check if the file is a PNG image
        if file.endswith('.png'):
            os.remove(full_file_path)

print("Image conversion from PNG to JPEG and deletion of original PNGs are complete.")
