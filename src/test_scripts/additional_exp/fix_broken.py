
import re
from pathlib import Path
import json

# -----------------------------
# Configuration
# -----------------------------
ROOT = Path("./")   # <-- change this to your root folder path


# -----------------------------
# Function to fix broken JSON content
# -----------------------------
def fix_json_text(text: str) -> str:
    """
    Fix JSON where objects are concatenated without commas.
    Example:  {}{}  →  {},{}
    """
    # Insert comma between `}{`
    fixed = re.sub(r'}\s*{', '},{', text)

    # Ensure list brackets exist
    fixed_stripped = fixed.strip()
    if not fixed_stripped.startswith("["):
        fixed = "[" + fixed
    if not fixed_stripped.endswith("]"):
        fixed = fixed + "]"

    return fixed


# -----------------------------
# Main repair process
# -----------------------------
def repair_all():
    print("Starting JSON repair...")

    # Loop over 4 main folders
    for main_folder in ROOT.iterdir():
        if not main_folder.is_dir():
            continue

        print(f"\nProcessing main folder: {main_folder.name}")

        # Find subfolders starting with "txt"
        for txt_folder in main_folder.iterdir():
            if not (txt_folder.is_dir() and txt_folder.name.startswith("txt")):
                continue

            print(f"  -> Repairing folder: {txt_folder.name}")

            # Find JSON files inside each txt folder
            json_files = list(txt_folder.glob("*.json"))

            if len(json_files) != 8:
                print(f"     WARNING: Expected 8 files, found {len(json_files)}. Still processing.")
            
            # Fix each JSON file
            for jf in json_files:
                try:
                    original_text = jf.read_text(encoding="utf-8")
                    fixed_text = fix_json_text(original_text)

                    # Save repaired content
                    out_path = jf.with_suffix(".fixed.json")
                    out_path.write_text(fixed_text, encoding="utf-8")

                    # Optional: test validity
                    json.loads(fixed_text)

                    print(f"     Repaired: {jf.name} -> {out_path.name}")

                except Exception as e:
                    print(f"     ERROR repairing {jf.name}: {e}")

    print("\nAll repairs completed successfully.")


# -----------------------------
# Run the script
# -----------------------------
if __name__ == "__main__":
    repair_all()
