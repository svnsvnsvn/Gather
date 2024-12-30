import json

# Load the annotations.json file
with open("data/raw/annotations.json", "r") as f:
    annotations = json.load(f)

# Inspect top-level keys
print("Top-level keys in the annotations file:", annotations.keys())

# Explore categories
categories = annotations["categories"]
print(f"Number of categories: {len(categories)}")
print("Categories:", [category["name"] for category in categories])

# Explore images and annotations
images = annotations["images"]
annotations_list = annotations["annotations"]
print(f"Number of images: {len(images)}")
print(f"Number of annotations: {len(annotations_list)}")


from collections import Counter

# Count annotations per category
category_counts = Counter([ann["category_id"] for ann in annotations_list])

# Map category IDs to names
category_map = {cat["id"]: cat["name"] for cat in categories}

# Print class distribution
for category_id, count in category_counts.items():
    print(f"Category: {category_map[category_id]}, Count: {count}")

# Optional: Visualize class distribution as a bar chart
import matplotlib.pyplot as plt

labels = [category_map[cat_id] for cat_id in category_counts.keys()]
values = list(category_counts.values())

plt.barh(labels, values)
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.savefig('test.png')
