import os
import json
from collections import defaultdict

# Task 1: Define the root directory containing the batches
BATCH_DIR = "../data/raw"

# Initialize counters
# category_counts will count how many times each category appears across all batches.
# category_batches will track which batch(es) contain a specific category.
category_counts = defaultdict(int)
category_batches = defaultdict(set)

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
        category_id = annotation['category_id']  
        category_name = next(
            (c['name'] for c in annotations['categories'] if c['id'] == category_id),  # Look up the category name
            None
        )

        if category_name:
            # Increment the count for this category
            category_counts[category_name] += 1
            # Record which batch contains this category
            category_batches[category_name].add(batch_name)

# Task 5: Identify underrepresented categories
THRESHOLD = 200  # Min number of samples for a category to be considered balanced
print(f"Underrepresented categories (fewer than {THRESHOLD} samples):")

underrep = dict()

for category, count in category_counts.items():
    if count < THRESHOLD:
        underrep[category] = count
        # print(f"{category}: {count} samples (Found in batches: {category_batches[category]})")

# Task 6: Save the analysis results
# OUTPUT_PATH: Path to save the analysis results (e.g., 'data/category_analysis.json')
OUTPUT_PATH = "../outputs/category_analysis.json"
with open(OUTPUT_PATH, 'w') as f:
    json.dump(
        {
            "counts": category_counts,  # Total counts for each category
            "batches": {k: list(v) for k, v in category_batches.items()},  # Batches for each category
            "underrepresented": underrep
        },
        f,
        indent=4
    )


print(f"Category analysis saved to {OUTPUT_PATH}")
