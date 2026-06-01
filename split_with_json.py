import os
import random
import shutil

# SOURCE FOLDER
source_folder = r"C:\Users\monic\OneDrive\Desktop\socila media\extracted_frames"

# OUTPUT DATASET FOLDER
dataset_folder = r"C:\Users\monic\OneDrive\Desktop\socila media\dataset"

# Create folders
splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(os.path.join(dataset_folder, split), exist_ok=True)

# Get all jpg images
images = [f for f in os.listdir(source_folder)
          if f.endswith(".jpg")]

# Shuffle images
random.shuffle(images)

# Split sizes
total = len(images)

train_end = int(0.7 * total)
val_end = int(0.85 * total)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

split_data = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

# Copy image + json
for split, image_list in split_data.items():

    print(f"\nProcessing {split}...")

    for image in image_list:

        image_src = os.path.join(source_folder, image)

        json_name = image.replace(".jpg", ".json")
        json_src = os.path.join(source_folder, json_name)

        image_dst = os.path.join(dataset_folder, split, image)
        json_dst = os.path.join(dataset_folder, split, json_name)

        # Copy image
        shutil.copy(image_src, image_dst)

        # Copy json if exists
        if os.path.exists(json_src):
            shutil.copy(json_src, json_dst)

    print(f"{split} completed")

print("\nALL FILES SPLIT SUCCESSFULLY")