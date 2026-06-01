import os
import glob

# dataset main folder
dataset_path = r"dataset"

# folders to check
folders = ["train", "val", "test"]

for folder in folders:
    path = os.path.join(dataset_path, folder)

    # find all json files
    json_files = glob.glob(os.path.join(path, "*.json"))

    for file in json_files:
        os.remove(file)
        print(f"Deleted: {file}")

print("All JSON files removed successfully!")