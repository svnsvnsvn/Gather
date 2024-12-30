import os
import json
from tqdm import tqdm

# Define the root directory for annotations
root_dir = '../data/raw'

# Define the refined category mapping
category_mapping = {
    "Clear plastic bottle": {"level_1": "Plastic", "level_2": "PET"},
    "Other plastic bottle": {"level_1": "Plastic", "level_2": "PET"},
    "Plastic bottle cap": {"level_1": "Plastic", "level_2": "PP"},
    "Plastic lid": {"level_1": "Plastic", "level_2": "PP"},
    "Polypropylene bag": {"level_1": "Plastic", "level_2": "PP"},
    "Disposable plastic cup": {"level_1": "Plastic", "level_2": "PP"},
    "Tupperware": {"level_1": "Plastic", "level_2": "PP"},
    "Plastic film": {"level_1": "Plastic", "level_2": "LDPE"},
    "Garbage bag": {"level_1": "Plastic", "level_2": "LDPE"},
    "Single-use carrier bag": {"level_1": "Plastic", "level_2": "LDPE"},
    "Six pack rings": {"level_1": "Plastic", "level_2": "LDPE"},
    "Squeezable tube": {"level_1": "Plastic", "level_2": "LDPE"},
    "Plastic utensils": {"level_1": "Plastic", "level_2": "PS"},
    "Plastic straw": {"level_1": "Plastic", "level_2": "PS"},
    "Foam cup": {"level_1": "Plastic", "level_2": "PS"},
    "Foam food container": {"level_1": "Plastic", "level_2": "PS"},
    "Styrofoam piece": {"level_1": "Plastic", "level_2": "PS"},
    "Other plastic wrapper": {"level_1": "Plastic", "level_2": "HDPE"},
    "Other plastic": {"level_1": "Plastic", "level_2": "HDPE"},
    "Aluminium foil": {"level_1": "Aluminium", "level_2": None},
    "Aluminium blister pack": {"level_1": "Aluminium", "level_2": None},
    "Drink can": {"level_1": "Aluminium", "level_2": None},
    "Food can": {"level_1": "Aluminium", "level_2": None},
    "Aerosol": {"level_1": "Aluminium", "level_2": None},
    "Corrugated carton": {"level_1": "Cardboard", "level_2": None},
    "Drink carton": {"level_1": "Cardboard", "level_2": None},
    "Egg carton": {"level_1": "Cardboard", "level_2": None},
    "Other carton": {"level_1": "Cardboard", "level_2": None},
    "Meal carton": {"level_1": "Cardboard", "level_2": None},
    "Pizza box": {"level_1": "Cardboard", "level_2": None},
    "Battery": {"level_1": "E-waste", "level_2": None},
    "Unlabeled litter": {"level_1": "E-waste", "level_2": None},
    "Cigarette": {"level_1": "Hazardous", "level_2": None},
    "Broken glass": {"level_1": "Glass", "level_2": None},
    "Glass bottle": {"level_1": "Glass", "level_2": None},
    "Glass jar": {"level_1": "Glass", "level_2": None},
    "Glass cup": {"level_1": "Glass", "level_2": None},
    "Paper cup": {"level_1": "Paper", "level_2": None},
    "Magazine paper": {"level_1": "Paper", "level_2": None},
    "Normal paper": {"level_1": "Paper", "level_2": None},
    "Tissues": {"level_1": "Paper", "level_2": None},
    "Wrapping paper": {"level_1": "Paper", "level_2": None},
    "Paper bag": {"level_1": "Paper", "level_2": None},
    "Plastified paper bag": {"level_1": "Paper", "level_2": None},
    "Paper straw": {"level_1": "Paper", "level_2": None},
    "Metal bottle cap": {"level_1": "Metal", "level_2": None},
    "Metal lid": {"level_1": "Metal", "level_2": None},
    "Pop tab": {"level_1": "Metal", "level_2": None},
    "Scrap metal": {"level_1": "Metal", "level_2": None},
    "Rope & strings": {"level_1": "Miscellaneous", "level_2": None},
    "Shoe": {"level_1": "Miscellaneous", "level_2": None},
    "Food waste": {"level_1": "Organic", "level_2": None},
    "Crisp packet": {"level_1": "Composite", "level_2": None},
    "Spread tub": {"level_1": "Composite", "level_2": None},
}

# Traverse all subdirectories and process each JSON annotation file
for root, _, files in os.walk(root_dir):
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(root, file_name)
            print(f"Processing {file_path}...")
            
            # Load the JSON annotation file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Update annotations
            for annotation in tqdm(data['annotations']):
                category_id = annotation['category_id']
                category_name = next((c['name'] for c in data['categories'] if c['id'] == category_id), None)

                if category_name in category_mapping:
                    annotation['level_1'] = category_mapping[category_name]['level_1']
                    annotation['level_2'] = category_mapping[category_name]['level_2']
                else:
                    annotation['level_1'] = None
                    annotation['level_2'] = None

            # Save the updated JSON back to the same file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"Updated annotations saved to {file_path}")
