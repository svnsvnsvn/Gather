"""
    Goal:
        * Split the dataset into training, validation, and test sets.
        * Augment images for the underrepresented categories in the training set only.
        * Save the augmented images in a structured format for future use.
"""

import os
import json
import shutil
import cv2
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

# Task 1: Split the dataset
BATCH_DIR = "../data/raw"  # Path to your batch directories
OUTPUT_SPLIT_DIR = "../data/split"  # Path to save split datasets
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)

# Create subdirectories for train, validation, and test
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_SPLIT_DIR, split), exist_ok=True)

# Task 1.1: Load annotations and map image paths to categories
image_paths = []
class_labels = []
for batch_name in os.listdir(BATCH_DIR):
    batch_path = os.path.join(BATCH_DIR, batch_name)
    annotation_file = os.path.join(batch_path, "annotations.json")
    if not os.path.exists(annotation_file):
        continue

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Create a mapping of image_id to category names
    image_category_map = {}  # Dictionary to map image_id to category names
    for ann in annotations['annotations']:
        category_id = ann['category_id']
        image_id = ann['image_id']

        # Look up the category name using the category_id
        category_name = next(
            (cat['name'] for cat in annotations['categories'] if cat['id'] == category_id),
            None
        )
        if category_name:
            # Add category to the mapping
            if image_id not in image_category_map:
                image_category_map[image_id] = set()
            image_category_map[image_id].add(category_name)

    # Map image paths to their categories
    for image in annotations['images']:
        image_id = image['id']
        file_name = image['file_name']

        # Get the categories associated with this image ID
        categories = image_category_map.get(image_id, set())
        for category in categories:
            image_paths.append(os.path.join(batch_path, file_name))
            class_labels.append(category)


# Task 1.2: Split the dataset into train, validation, and test sets
"""
    the train_test_split function from scikit splits the dataset into 2 subsets while maintaining the proportions of the class labels (stratify ensures this)

    * image_paths: a list of all the file paths to the images 
    * class_labels: a list of labels (i.e, plastic, cardboard) corresponding to each image in image paths

    * train_paths and train_labels = 80% of the data used for training 
    * temp_paths and temp_labels = 20% of the data used for splitting further into validation and test sets. 
"""

from collections import Counter

# Count occurrences of each class
label_counts = Counter(class_labels)
# print("Class Distribution:", label_counts)

# Check for classes with fewer than 2 samples
# few_samples = {label: count for label, count in label_counts.items() if count < 2}
# if few_samples:
#     print("Classes with too few samples:", few_samples)

# Filter out classes with fewer than 2 counts
valid_labels = {label for label, count in label_counts.items() if count >= 2}
filtered_image_path = [
    img_path for img_path, label in zip(image_paths, class_labels) if label in valid_labels
]

filtered_class_labels = [
    label for label in class_labels if label in valid_labels
]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filtered_image_path, filtered_class_labels, test_size=0.2, stratify=filtered_class_labels, random_state=42
)

# Identify Rare Classes
RARE_THRESHOLD = 2  # Classes with fewer than 2 samples will be combined
temp_label_counts = Counter(temp_labels)

# Map rare classes to "Other"
combined_labels = [
    label if temp_label_counts[label] >= RARE_THRESHOLD else "Other"
    for label in temp_labels
]

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, combined_labels, test_size=0.5, stratify=combined_labels, random_state=42
)

# Task 1.3: Move images to respective directories
for split, paths in zip(["train", "validation", "test"], [train_paths, val_paths, test_paths]):
    for image_path in paths:
        dst = os.path.join(OUTPUT_SPLIT_DIR, split, os.path.basename(image_path))
        shutil.copy(image_path, dst)

print(f"Dataset split completed. Check {OUTPUT_SPLIT_DIR}")

# Task 1.4: Create annotations for each split
for split, paths in zip(["train", "validation", "test"], [train_paths, val_paths, test_paths]):
    split_images = []
    split_annotations = []

    # Filter images and annotations for the current split
    for image_path in paths:
        file_name = os.path.basename(image_path)
        # Find the image in the original annotations
        image_entry = next((img for img in annotations['images'] if img['file_name'] == file_name), None)
        if image_entry:
            split_images.append(image_entry)
            # Find corresponding annotations
            image_id = image_entry['id']
            split_annotations.extend([ann for ann in annotations['annotations'] if ann['image_id'] == image_id])

    # Create a new annotation file for the split
    split_annotation = {
        'images': split_images,  # Only images in the split
        'annotations': split_annotations,  # Corresponding annotations
        # Add other keys if they exist in your annotation structure
    }

    # Save the annotations to the split directory
    split_annotation_path = os.path.join(OUTPUT_SPLIT_DIR, split, 'annotations.json')
    with open(split_annotation_path, 'w') as f:
        json.dump(split_annotation, f, indent=4)

    print(f"Annotations added to {split_annotation_path}")


# Task 2: Load the category analysis
CATEGORY_ANALYSIS_FILE = "../outputs/category_analysis.json"  # Path to 'category_analysis.json'
with open(CATEGORY_ANALYSIS_FILE, 'r') as f:
    category_analysis = json.load(f)

underrepresented = category_analysis["underrepresented"]  # Categories needing augmentation
category_batches = category_analysis["batches"]  # Map of categories to batches

# Task 3: Define the augmentation pipeline
augment = A.Compose([
    A.HorizontalFlip(p=0.5),  # flip images horizontally
    A.Rotate(limit=30, p=0.5),  # rotate images randomly up to 30 degrees
    A.RandomBrightnessContrast(p=0.5),  # adjust brightness and contrast
    A.Perspective(scale=(0.05, 0.1), p=0.5),  # apply perspective transformations
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # add blur to simulate camera imperfections
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # color adjustments
    A.RandomCrop(height=200, width=200, p=0.5),  # randomly crop parts of the image
    A.Resize(224, 224)  # resize to 224x224
])


# Task 4: Apply augmentation to training set only
train_dir = os.path.join(OUTPUT_SPLIT_DIR, "train")
augmented_dir = os.path.join(train_dir, "augmented")
os.makedirs(augmented_dir, exist_ok=True)

TARGET_COUNT = 200 # minimum number of images per category

for category, count in tqdm(underrepresented.items(), desc="Processing categories"):
    print(f"Processing category: {category} (Count: {count})")

    # Calculate how many images to augment
    num_to_augment = max(0, TARGET_COUNT - count)

    # Task 4.1: Locate images for the current category in the training set
    category_images = [
        img for img in os.listdir(train_dir)
        if img.endswith(('.jpg', '.jpeg', '.png')) and category in category_batches
    ]

    if not category_images:
        print(f"No images found for category {category}. Skipping...")
        continue

    for i in range(num_to_augment):
        # Cycle through available images
        original_image_name = category_images[i % len(category_images)]
        image_path = os.path.join(train_dir, original_image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}. Skipping...")
            continue

        # Task 4.2: Load and augment the image
        augmented = augment(image=image)

        # Save augmented image
        aug_image_name = f"aug_{i}_{original_image_name}"  # Unique name for each augmented image
        aug_image_path = os.path.join(augmented_dir, aug_image_name)
        cv2.imwrite(aug_image_path, augmented['image'])
        # print(f"Augmented image saved to {aug_image_path}")