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
from albumentations.core.composition import OneOf
# from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes

from albumentations.core.utils import format_args
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
        'categories': annotations['categories'],
        'scene_categories': annotations.get('scene_categories', [])
    }

    # Save the annotations to the split directory
    split_annotation_path = os.path.join(OUTPUT_SPLIT_DIR, split, 'annotations.json')
    with open(split_annotation_path, 'w') as f:
        json.dump(split_annotation, f, indent=4)

    print(f"Annotations added to {split_annotation_path}")

# Normalize bounding boxes to [0.0, 1.0]
def normalize_bbox(bbox, image_width, image_height):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    # Normalize values
    normalized_bbox = [
        max(0.0, min(x_min / image_width, 1.0)),
        max(0.0, min(y_min / image_height, 1.0)),
        max(0.0, min(x_max / image_width, 1.0)),
        max(0.0, min(y_max / image_height, 1.0))
    ]
    return normalized_bbox

def validate_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    return 0.0 <= x_min <= 1.0 and 0.0 <= y_min <= 1.0 and 0.0 <= x_max <= 1.0 and 0.0 <= y_max <= 1.0

def clamp_bbox(bbox, width, height):
    """
    Ensure that the bounding box coordinates are within the image boundaries.
    """
    x_min = max(0, min(bbox[0], width))
    y_min = max(0, min(bbox[1], height))
    x_max = max(0, min(bbox[0] + bbox[2], width))
    y_max = max(0, min(bbox[1] + bbox[3], height))

    clamped_width = max(0, x_max - x_min)
    clamped_height = max(0, y_max - y_min)

    return [x_min, y_min, clamped_width, clamped_height]


# Denormalize bounding boxes back to absolute pixel values
def denormalize_bbox(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    return [
        x_min * image_width,
        y_min * image_height,
        (x_max - x_min) * image_width,
        (y_max - y_min) * image_height,
    ]

# Function to normalize segmentation points
def normalize_segmentation(segmentation, width, height):
    """
    Normalize segmentation coordinates to [0, 1].
    Each segmentation is a list of polygons, and this function processes each polygon.
    """
    normalized_seg = []
    for polygon in segmentation:  # Iterate over each polygon
        normalized_polygon = [
            coord / width if i % 2 == 0 else coord / height
            for i, coord in enumerate(polygon)
        ]
        normalized_seg.append(normalized_polygon)
    return normalized_seg

def denormalize_segmentation(segmentation, width, height):
    """
    Denormalize segmentation coordinates from [0, 1] back to pixel values.
    """
    denormalized_seg = []
    for polygon in segmentation:
        denormalized_polygon = [
            coord * width if i % 2 == 0 else coord * height
            for i, coord in enumerate(polygon)
        ]
        denormalized_seg.append(denormalized_polygon)
    return denormalized_seg


# Task 2: Load the category analysis

# Paths
train_dir = os.path.join(OUTPUT_SPLIT_DIR, "train")
annotations_path = os.path.join(train_dir, "annotations.json")
augmented_dir = os.path.join(train_dir, "augmented")
os.makedirs(augmented_dir, exist_ok=True)

# Load primary training annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Include the categories and scene_categories in the final augmented_metadata
categories = [
    {
            "supercategory": "Aluminium foil",
            "id": 0,
            "name": "Aluminium foil"
        },
        {
            "supercategory": "Battery",
            "id": 1,
            "name": "Battery"
        },
        {
            "supercategory": "Blister pack",
            "id": 2,
            "name": "Aluminium blister pack"
        },
        {
            "supercategory": "Blister pack",
            "id": 3,
            "name": "Carded blister pack"
        },
        {
            "supercategory": "Bottle",
            "id": 4,
            "name": "Other plastic bottle"
        },
        {
            "supercategory": "Bottle",
            "id": 5,
            "name": "Clear plastic bottle"
        },
        {
            "supercategory": "Bottle",
            "id": 6,
            "name": "Glass bottle"
        },
        {
            "supercategory": "Bottle cap",
            "id": 7,
            "name": "Plastic bottle cap"
        },
        {
            "supercategory": "Bottle cap",
            "id": 8,
            "name": "Metal bottle cap"
        },
        {
            "supercategory": "Broken glass",
            "id": 9,
            "name": "Broken glass"
        },
        {
            "supercategory": "Can",
            "id": 10,
            "name": "Food Can"
        },
        {
            "supercategory": "Can",
            "id": 11,
            "name": "Aerosol"
        },
        {
            "supercategory": "Can",
            "id": 12,
            "name": "Drink can"
        },
        {
            "supercategory": "Carton",
            "id": 13,
            "name": "Toilet tube"
        },
        {
            "supercategory": "Carton",
            "id": 14,
            "name": "Other carton"
        },
        {
            "supercategory": "Carton",
            "id": 15,
            "name": "Egg carton"
        },
        {
            "supercategory": "Carton",
            "id": 16,
            "name": "Drink carton"
        },
        {
            "supercategory": "Carton",
            "id": 17,
            "name": "Corrugated carton"
        },
        {
            "supercategory": "Carton",
            "id": 18,
            "name": "Meal carton"
        },
        {
            "supercategory": "Carton",
            "id": 19,
            "name": "Pizza box"
        },
        {
            "supercategory": "Cup",
            "id": 20,
            "name": "Paper cup"
        },
        {
            "supercategory": "Cup",
            "id": 21,
            "name": "Disposable plastic cup"
        },
        {
            "supercategory": "Cup",
            "id": 22,
            "name": "Foam cup"
        },
        {
            "supercategory": "Cup",
            "id": 23,
            "name": "Glass cup"
        },
        {
            "supercategory": "Cup",
            "id": 24,
            "name": "Other plastic cup"
        },
        {
            "supercategory": "Food waste",
            "id": 25,
            "name": "Food waste"
        },
        {
            "supercategory": "Glass jar",
            "id": 26,
            "name": "Glass jar"
        },
        {
            "supercategory": "Lid",
            "id": 27,
            "name": "Plastic lid"
        },
        {
            "supercategory": "Lid",
            "id": 28,
            "name": "Metal lid"
        },
        {
            "supercategory": "Other plastic",
            "id": 29,
            "name": "Other plastic"
        },
        {
            "supercategory": "Paper",
            "id": 30,
            "name": "Magazine paper"
        },
        {
            "supercategory": "Paper",
            "id": 31,
            "name": "Tissues"
        },
        {
            "supercategory": "Paper",
            "id": 32,
            "name": "Wrapping paper"
        },
        {
            "supercategory": "Paper",
            "id": 33,
            "name": "Normal paper"
        },
        {
            "supercategory": "Paper bag",
            "id": 34,
            "name": "Paper bag"
        },
        {
            "supercategory": "Paper bag",
            "id": 35,
            "name": "Plastified paper bag"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 36,
            "name": "Plastic film"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 37,
            "name": "Six pack rings"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 38,
            "name": "Garbage bag"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 39,
            "name": "Other plastic wrapper"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 40,
            "name": "Single-use carrier bag"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 41,
            "name": "Polypropylene bag"
        },
        {
            "supercategory": "Plastic bag & wrapper",
            "id": 42,
            "name": "Crisp packet"
        },
        {
            "supercategory": "Plastic container",
            "id": 43,
            "name": "Spread tub"
        },
        {
            "supercategory": "Plastic container",
            "id": 44,
            "name": "Tupperware"
        },
        {
            "supercategory": "Plastic container",
            "id": 45,
            "name": "Disposable food container"
        },
        {
            "supercategory": "Plastic container",
            "id": 46,
            "name": "Foam food container"
        },
        {
            "supercategory": "Plastic container",
            "id": 47,
            "name": "Other plastic container"
        },
        {
            "supercategory": "Plastic glooves",
            "id": 48,
            "name": "Plastic glooves"
        },
        {
            "supercategory": "Plastic utensils",
            "id": 49,
            "name": "Plastic utensils"
        },
        {
            "supercategory": "Pop tab",
            "id": 50,
            "name": "Pop tab"
        },
        {
            "supercategory": "Rope & strings",
            "id": 51,
            "name": "Rope & strings"
        },
        {
            "supercategory": "Scrap metal",
            "id": 52,
            "name": "Scrap metal"
        },
        {
            "supercategory": "Shoe",
            "id": 53,
            "name": "Shoe"
        },
        {
            "supercategory": "Squeezable tube",
            "id": 54,
            "name": "Squeezable tube"
        },
        {
            "supercategory": "Straw",
            "id": 55,
            "name": "Plastic straw"
        },
        {
            "supercategory": "Straw",
            "id": 56,
            "name": "Paper straw"
        },
        {
            "supercategory": "Styrofoam piece",
            "id": 57,
            "name": "Styrofoam piece"
        },
        {
            "supercategory": "Unlabeled litter",
            "id": 58,
            "name": "Unlabeled litter"
        },
        {
            "supercategory": "Cigarette",
            "id": 59,
            "name": "Cigarette"
        }
]  


scene_categories = [
    {"id": 0, "name": "Clean"},
    {"id": 1, "name": "Indoor, Man-made"},
    {"id": 2, "name": "Pavement"},
    {"id": 3, "name": "Sand, Dirt, Pebbles"},
    {"id": 4, "name": "Trash"},
    {"id": 5, "name": "Vegetation"},
    {"id": 6, "name": "Water"}
]

scene_category_mapping = {
    "clean": 0,
    "indoor": 1,
    "pavement": 2,
    "sand": 3,
    "trash": 4,
    "vegetation": 5,
    "water": 6
}

# Assign scene_category_id during augmentation
# todo


# Prepare a separate file for augmented images metadata
augmented_metadata = {
    "images": [],
    "annotations": [],
    "categories": categories,
    "scene_categories": scene_categories
}

TARGET_COUNT = 200 # minimum number of images per category

CATEGORY_ANALYSIS_FILE = "../outputs/category_analysis.json"  # Path to 'category_analysis.json'
with open(CATEGORY_ANALYSIS_FILE, 'r') as f:
    category_analysis = json.load(f)

underrepresented = category_analysis["underrepresented"]  # Categories needing augmentation

augment = A.Compose([
    A.HorizontalFlip(p=0.5),  # flip images horizontally
    A.Rotate(limit=30, p=0.5),  # rotate images randomly up to 30 degrees
    A.RandomBrightnessContrast(p=0.5),  # adjust brightness and contrast
    A.Perspective(scale=(0.05, 0.1), p=0.5),  # apply perspective transformations
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # add blur to simulate camera imperfections
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),  # color adjustments
    A.RandomCrop(height=200, width=200, p=0.5),  # randomly crop parts of the image
    A.Resize(224, 224)  # resize to 224x224
],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),  # Transform bounding boxes
    # keypoint_params=A.KeypointParams(format='xy')  # Transform segmentation keypoints (if applicable)
)


for category_id, count in tqdm(underrepresented.items(), desc="Processing categories"):
    print(f"Processing category ID: {category_id} (Count: {count})")
    num_to_augment = max(0, TARGET_COUNT - count)

    category_images = [
        img["file_name"]
        for img in annotations["images"]
        if any(ann["category_id"] == int(category_id) for ann in annotations["annotations"] if ann["image_id"] == img["id"])
    ]

    if not category_images:
        print(f"No images found for category ID {category_id}. Skipping...")
        continue

    for i in range(num_to_augment):
        original_image_name = category_images[i % len(category_images)]
        image_path = os.path.join(train_dir, original_image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}. Skipping...")
            continue

        image_id = next(img["id"] for img in annotations["images"] if img["file_name"] == original_image_name)

        bboxes = [ann["bbox"] for ann in annotations["annotations"] if ann["image_id"] == image_id]
        category_ids = [ann["category_id"] for ann in annotations["annotations"] if ann["image_id"] == image_id]
        segmentations = [ann["segmentation"] for ann in annotations["annotations"] if ann["image_id"] == image_id]

        # Normalize bounding boxes and segmentation points
        normalized_bboxes = [normalize_bbox(bbox, image.shape[1], image.shape[0]) for bbox in bboxes]
        valid_bboxes = [bbox for bbox in normalized_bboxes if validate_bbox(bbox)]
        if len(valid_bboxes) != len(normalized_bboxes):
            print(f"Invalid bounding boxes found: {len(normalized_bboxes) - len(valid_bboxes)}")
        
        normalized_segmentations = [
            normalize_segmentation(seg, image.shape[1], image.shape[0]) for seg in segmentations
        ]

        # Apply augmentation
        augmented = augment(image=image, bboxes=valid_bboxes, category_ids=category_ids)

        # Denormalize augmented bounding boxes and segmentation points
        denormalized_bboxes = [
            denormalize_bbox(bbox, augmented["image"].shape[1], augmented["image"].shape[0])
            for bbox in augmented["bboxes"]
        ]
        denormalized_segmentations = [
            denormalize_segmentation(seg, augmented["image"].shape[1], augmented["image"].shape[0])
            for seg in normalized_segmentations
        ]

        aug_image_name = f"aug_{i}_{original_image_name}"
        aug_image_path = os.path.join(augmented_dir, aug_image_name)
        cv2.imwrite(aug_image_path, augmented['image'])

        new_image_id = max(img["id"] for img in annotations["images"]) + 1
        new_image_entry = {
            "id": new_image_id,
            "height": augmented["image"].shape[0],
            "width": augmented["image"].shape[1],
            "file_name": os.path.join(aug_image_name),

        }
        annotations["images"].append(new_image_entry)
        augmented_metadata["images"].append(new_image_entry)

        for bbox, category_id, segmentation in zip(
            denormalized_bboxes, augmented["category_ids"], denormalized_segmentations
        ):
            new_annotation = {
                "id": max(ann["id"] for ann in annotations["annotations"]) + 1,
                "image_id": new_image_id,
                "category_id": category_id,
                "bbox": bbox,
                "segmentation": segmentation,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            annotations["annotations"].append(new_annotation)
            augmented_metadata["annotations"].append(new_annotation)

# Append augmented annotations to the training file
annotations["annotations"].extend(augmented_metadata["annotations"])

# Save updated primary annotations
with open(annotations_path, "w") as f:
    json.dump(annotations, f, indent=4)

# Save augmented metadata
augmented_metadata_path = os.path.join(augmented_dir, "augmented_annotations.json")
with open(augmented_metadata_path, "w") as f:
    json.dump(augmented_metadata, f, indent=4)

# Issues:
# 1. image_id and annotation.image_id mismatch, leading to incorrect linkages.
# 2. Bounding boxes exceed image dimensions or have invalid (negative/zero) sizes due to augmentation issues.
# 3. Segmentation points go out of bounds or are invalid (e.g., fewer than 3 points).
# 4. Negative or zero area values indicate invalid bounding boxes or segmentations.
# 5. Underrepresented categories might not be augmented adequately due to incorrect image selection.
# 6. Augmented metadata (images, annotations) may not align with COCO format.
# 7. Visualization fails due to invalid bounding boxes, segmentations, or missing image paths.
# 8. Validation checks may miss inconsistencies in bounding boxes, segmentations, or area calculations.
