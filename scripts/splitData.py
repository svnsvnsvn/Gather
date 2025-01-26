import os
import json
import shutil
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

BATCH_DIR = "../data/raw"

OUTPUT_SPLIT_DIR = "../data/split"
os.makedirs(OUTPUT_SPLIT_DIR, exist_ok=True)
for split_name in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_SPLIT_DIR, split_name), exist_ok=True)

# Global ID counters
next_image_id = 1
next_annotation_id = 1

batch_annotations = {}
image_paths = []
class_labels = []


subfolders = sorted(os.listdir(BATCH_DIR))  # e.g. ["batch_1","batch_2","batch_12",...]
for subfolder in tqdm(subfolders, desc="Processing batches"):
    batch_path = os.path.join(BATCH_DIR, subfolder)
    
    # Skip if not actually a directory (some systems have hidden files)
    if not os.path.isdir(batch_path):
        continue
    
    annotation_file = os.path.join(batch_path, "annotations.json")
    if not os.path.exists(annotation_file):
        print(f"Annotation file missing in {subfolder}, skipping.")
        continue
    
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    
    # convert local IDs to global unique IDs
    id_mapping = {}
    
    # Re-label images
    for img in annotations["images"]:
        old_id = img["id"]
        new_id = next_image_id
        id_mapping[old_id] = new_id
        
        img["id"] = new_id
        # If subfolder is "batch_12", final name e.g. "batch_12_000040.jpg"
        img["file_name"] = f"{subfolder}_{img['file_name']}"
        
        next_image_id += 1
    
    # Re-label annotations
    for ann in annotations["annotations"]:
        ann["id"] = next_annotation_id
        ann["image_id"] = id_mapping[ann["image_id"]]
        next_annotation_id += 1
    
    # Map images => categories
    image_category_map = {}
    for ann in annotations["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        cat_name = next((c["name"] for c in annotations["categories"] if c["id"] == cat_id), None)
        if cat_name:
            image_category_map.setdefault(img_id, set()).add(cat_name)
    
    # Build global list for train_test_split
    for img in annotations["images"]:
        img_id = img["id"]
        final_filename = img["file_name"]  
        
        original_filename = final_filename.split("_", 1)[1]  # remove the subfolder prefix
        raw_path = os.path.join(BATCH_DIR, subfolder, original_filename)
        
        categories = image_category_map.get(img_id, [])
        for cat in categories:
            image_paths.append(raw_path)
            class_labels.append(cat)
    
    # Store updated annotations in memory
    batch_annotations[subfolder] = annotations

print(f"Collected {len(image_paths)} labeled images total.")

label_counts = Counter(class_labels)
valid_labels = {l for l, c in label_counts.items() if c >= 2}
filtered_image_paths = []
filtered_class_labels = []
for p, l in zip(image_paths, class_labels):
    if l in valid_labels:
        filtered_image_paths.append(p)
        filtered_class_labels.append(l)
print(f"After filtering, {len(filtered_image_paths)} remain.")


train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filtered_image_paths,
    filtered_class_labels,
    test_size=0.2,
    stratify=filtered_class_labels,
    random_state=42
)

temp_label_counts = Counter(temp_labels)
def maybe_other(label):
    return label if temp_label_counts[label] >= 2 else "Other"

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    [maybe_other(l) for l in temp_labels],
    test_size=0.5,
    stratify=[maybe_other(l) for l in temp_labels],
    random_state=42
)

print("Split sizes:")
print(f"  Train: {len(train_paths)}")
print(f"  Val:   {len(val_paths)}")
print(f"  Test:  {len(test_paths)}")


split_annotations = {
    "train": {"images": [], "annotations": [], "categories": []},
    "validation": {"images": [], "annotations": [], "categories": []},
    "test": {"images": [], "annotations": [], "categories": []},
}

added_image_ids = {s: set() for s in split_annotations}
added_annotation_ids = {s: set() for s in split_annotations}
global_assigned_images = set()


for subfolder, ann_data in tqdm(batch_annotations.items(), desc="Assigning to splits"):
    # Filter each split's paths for the current subfolder
    train_paths_batch = [p for p in train_paths if os.path.dirname(p).endswith(subfolder)]
    val_paths_batch   = [p for p in val_paths   if os.path.dirname(p).endswith(subfolder)]
    test_paths_batch  = [p for p in test_paths  if os.path.dirname(p).endswith(subfolder)]
    
    # Ensure categories appear if not yet set
    if not split_annotations["train"]["categories"]:
        split_annotations["train"]["categories"] = ann_data["categories"]
    if not split_annotations["validation"]["categories"]:
        split_annotations["validation"]["categories"] = ann_data["categories"]
    if not split_annotations["test"]["categories"]:
        split_annotations["test"]["categories"] = ann_data["categories"]
    
    # Assign to each of the three splits
    for split_name, batch_paths in zip(["train", "validation", "test"],
                                       [train_paths_batch, val_paths_batch, test_paths_batch]):

        for raw_path in batch_paths:
            # raw_path is e.g. ../data/raw/batch_12/000040.jpg
            file_name_only = os.path.basename(raw_path)  
            prefixed_name = f"{subfolder}_{file_name_only}"  

            if prefixed_name in global_assigned_images:
                continue
            global_assigned_images.add(prefixed_name)

            # Find matching image entry
            img_entry = next(
                (img for img in ann_data["images"] if img["file_name"].endswith(file_name_only)),
                None
            )
            if not img_entry:
                # Possibly extension mismatch or not in this batch
                continue

            if img_entry["id"] not in added_image_ids[split_name]:
                split_annotations[split_name]["images"].append(img_entry)
                added_image_ids[split_name].add(img_entry["id"])

            for an in ann_data["annotations"]:
                if an["image_id"] == img_entry["id"] and an["id"] not in added_annotation_ids[split_name]:
                    split_annotations[split_name]["annotations"].append(an)
                    added_annotation_ids[split_name].add(an["id"])


for split_name in ["train", "validation", "test"]:
    split_dir = os.path.join(OUTPUT_SPLIT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Copy images
    for img_info in split_annotations[split_name]["images"]:

        first_part, rest = img_info["file_name"].split("_", 1)
        sub_part, original_filename = rest.split("_", 1)
        subfolder = first_part + "_" + sub_part

        src = os.path.join(BATCH_DIR, subfolder, original_filename)
        dst = os.path.join(split_dir, img_info["file_name"])

        # fallback for extension mismatch
        if not os.path.exists(src):
            alt_src = os.path.splitext(src)[0] + ".JPG"
            if os.path.exists(alt_src):
                src = alt_src

        if not os.path.exists(src):
            print(f"WARNING: source not found: {src}")
            continue

        shutil.copy(src, dst)

    # Save annotations
    anno_path = os.path.join(split_dir, "annotations.json")
    with open(anno_path, "w") as f:
        json.dump(split_annotations[split_name], f, indent=4)


for split_name in ["train", "validation", "test"]:
    n_imgs = len(split_annotations[split_name]["images"])
    n_anns = len(split_annotations[split_name]["annotations"])
    print(f"\n{split_name.capitalize()} Split:")
    print(f"  Images: {n_imgs}")
    print(f"  Annotations: {n_anns}")

print(f"\nDone! Check {OUTPUT_SPLIT_DIR} for results. :)")
