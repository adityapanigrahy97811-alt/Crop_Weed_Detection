import os
import random
import shutil

# Path to your dataset folder (where all images and txt files are now)
base_path = "dataset"

# Create required folders
folders = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Get all image files
all_files = os.listdir(base_path)
images = [f for f in all_files if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")]

# Shuffle randomly
random.shuffle(images)

# Split 80% train, 20% val
split_index = int(len(images) * 0.8)
train_images = images[:split_index]
val_images = images[split_index:]

# Function to move files
def move_files(image_list, split_type):
    for img in image_list:
        label = img.rsplit(".", 1)[0] + ".txt"

        shutil.move(os.path.join(base_path, img),
                    os.path.join(base_path, "images", split_type, img))

        if os.path.exists(os.path.join(base_path, label)):
            shutil.move(os.path.join(base_path, label),
                        os.path.join(base_path, "labels", split_type, label))

# Move files
move_files(train_images, "train")
move_files(val_images, "val")

print("Dataset split completed successfully!")


