#!/usr/bin/env python3
import os
from pathlib import Path

TARGET_DIR = Path("experiment_outputs")
OLD_PATH = "[LOCAL_AUTHOR_PATH]/OpenRE-Bench"
NEW_PATH = "[ANONYMIZED_DIR]/OpenRE-Bench"

def anonymize_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if OLD_PATH in content:
            new_content = content.replace(OLD_PATH, NEW_PATH)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            # print(f"Anonymized: {filepath}")
            return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return False

def main():
    if not TARGET_DIR.exists():
        print(f"Directory {TARGET_DIR} not found.")
        return

    count = 0
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(('.json', '.jsonl', '.md', '.csv')):
                if anonymize_file(Path(root) / file):
                    count += 1
                    
    print(f"Anonymized {count} files in {TARGET_DIR}.")

if __name__ == "__main__":
    main()
