import json
import os

# CLASS NAMES
classes = {
    "phone": 0
}

# DATASET PATH
dataset_path = r"C:\Users\monic\OneDrive\Desktop\socila media\dataset"

# Folders
splits = ["train", "val", "test"]

for split in splits:

    split_path = os.path.join(dataset_path, split)

    json_files = [f for f in os.listdir(split_path)
                  if f.endswith(".json")]

    print(f"\nProcessing {split}...")

    for json_file in json_files:

        json_path = os.path.join(split_path, json_file)

        with open(json_path, "r") as f:
            data = json.load(f)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        txt_filename = json_file.replace(".json", ".txt")
        txt_path = os.path.join(split_path, txt_filename)

        with open(txt_path, "w") as txt_file:

            for shape in data["shapes"]:

                label = shape["label"]

                # Skip unknown labels
                if label not in classes:
                    continue

                class_id = classes[label]

                points = shape["points"]

                normalized_points = []

                for point in points:

                    x = point[0] / image_width
                    y = point[1] / image_height

                    normalized_points.append(f"{x:.6f}")
                    normalized_points.append(f"{y:.6f}")

                line = f"{class_id} " + " ".join(normalized_points)

                txt_file.write(line + "\n")

        print(f"Converted: {json_file}")

print("\nALL JSON FILES CONVERTED")