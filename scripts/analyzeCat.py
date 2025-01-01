import os
import json
from collections import defaultdict

# Task 1: Define the root directory containing the batches
BATCH_DIR = "../data/raw"

# Initialize counters
category_counts = defaultdict(int)  # Count categories by category_id
category_batches = defaultdict(set)  # Track batches containing each category_id

# Task 2: Loop through all batch directories
for batch_name in os.listdir(BATCH_DIR):
    batch_path = os.path.join(BATCH_DIR, batch_name)
    if not os.path.isdir(batch_path): 
        continue

    # Task 2.1: Locate the annotations.json file for this batch
    annotation_path = os.path.join(batch_path, "annotations.json")
    if not os.path.exists(annotation_path):  # Skip batches without an annotations file
        print(f"No annotations found in {batch_path}. Skipping...")
        continue

    # Task 3: Load the annotations.json file
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Task 4: Count categories and track batches
    for annotation in annotations['annotations']:
        category_id = annotation['category_id']  # Use category_id directly

        # Increment the count for this category_id
        category_counts[category_id] += 1
        # Record which batch contains this category_id
        category_batches[category_id].add(batch_name)

# Task 5: Identify underrepresented categories
THRESHOLD = 200  # Min number of samples for a category to be considered balanced
print(f"Underrepresented categories (fewer than {THRESHOLD} samples):")

underrep = {
    category_id: count
    for category_id, count in category_counts.items()
    if count < THRESHOLD
}

# Task 6: Save the analysis results
OUTPUT_PATH = "../outputs/category_analysis.json"
with open(OUTPUT_PATH, 'w') as f:
    json.dump(
        {
            "counts": dict(category_counts),  # Convert defaultdict to regular dict
            "batches": {k: list(v) for k, v in category_batches.items()},  # Convert sets to lists
            "underrepresented": underrep  # Use IDs for underrepresented categories
        },
        f,
        indent=4
    )

print(f"Category analysis saved to {OUTPUT_PATH}")
