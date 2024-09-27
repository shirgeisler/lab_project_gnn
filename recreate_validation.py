import os
import shutil

# Paths to validation images and annotation file
val_img_dir = 'data/tiny-imagenet-200/val/images/'
val_annotations = 'data/tiny-imagenet-200/val/val_annotations.txt'
val_output_dir = 'data/tiny-imagenet-200/val/'

# Create output directory if it doesn't exist
if not os.path.exists(val_output_dir):
    os.makedirs(val_output_dir)

# Read the annotations file and organize images into subfolders
with open(val_annotations, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split()
        img_name, label = parts[0], parts[1]
        label_dir = os.path.join(val_output_dir, label)

        # Create directory for the label if it doesn't exist
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Move the image to the corresponding label directory
        src_img_path = os.path.join(val_img_dir, img_name)
        dest_img_path = os.path.join(label_dir, img_name)
        shutil.move(src_img_path, dest_img_path)

print("Validation images organized into subfolders.")