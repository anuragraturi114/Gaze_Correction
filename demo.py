import os
import shutil

# Define source and destination directories
source_dirs = ["/Users/anuragraturi/PycharmProjects/detection/Extracted Faces/Extracted Faces", "/Users/anuragraturi/PycharmProjects/detection/Face Data/Face Dataset"]
destination_dir = "/Users/anuragraturi/PycharmProjects/detection/images"

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Merge data from source directories into destination directory
for source_dir in source_dirs:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Copy files to destination directory
            shutil.copy(os.path.join(root, file), destination_dir)

