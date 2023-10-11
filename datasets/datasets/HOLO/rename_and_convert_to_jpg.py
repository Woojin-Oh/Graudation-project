import os
from PIL import Image

# Define the directory where your images are located
base_dir = '/home/an258456/Real-time-GesRec/datasets/HOLO'

# Initialize the counter for image names


# Iterate over all subdirectories and files in your base directory
for root, dirs, files in os.walk(base_dir):
    counter = 1
    for file in files:
        # Check if the file is a PNG image
        if file.endswith('.png'):
            # Get the full path of the source image and where to save it after renaming 
            full_file_path = os.path.join(root, file)
            print('current path: ', full_file_path)
            new_file_name = f"{str(counter).zfill(5)}.jpg"
            new_file_path = os.path.join(root, new_file_name)
            
            # Open the PNG image using PIL
            img = Image.open(full_file_path)
            
            # Convert the image to RGB mode (remove alpha channel if exists)
            img = img.convert('RGB')
            
            # Save the converted image as JPEG format with quality 95 (adjust as needed)
            img.save(new_file_path, 'JPEG', quality=100)

            counter +=1

print("Image renaming is complete.")