import os
import json
from pathlib import Path

ROOT = Path("./")   # change to your top-level folder

# -----------------------------
# Function to merge JSON files
# -----------------------------
def merge_json_files(json_files):
    merged = []

    for f in json_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            # If json file is a list → extend
            if isinstance(data, list):
                merged.extend(data)
            # If json file is a dict → append dict
            else:
                merged.append(data)

    return merged


# -----------------------------
# Walk through 4 root folders
# -----------------------------
for folder in ROOT.iterdir():
    if folder.is_dir():
        print(f"\nProcessing top folder: {folder.name}")

        # ===========================================
        # For each of the 6 subfolders inside this folder:
        # Create a new folder:  txt_<old_subfolder_name>
        # ===========================================
        for subfolder in folder.iterdir():
            if subfolder.is_dir():

                # 1) Create new folder with prefix
                new_folder_name = f"txt_{subfolder.name}"
                new_folder_path = folder / new_folder_name
                new_folder_path.mkdir(exist_ok=True)

                print(f"  Created: {new_folder_path}")

                # 2) Merge JSON files inside old subfolder (if exactly 8 json files exist)
                json_files = list(subfolder.glob("*.json"))
                if len(json_files) == 8:
                    print(f"  → Merging JSON files in: {subfolder.name}")

                    merged_data = merge_json_files(json_files)

                    output_path = subfolder / "merged.json"
                    with open(output_path, "w", encoding="utf-8") as out:
                        json.dump(merged_data, out, indent=2, ensure_ascii=False)

print("\n✓ All tasks done.")
