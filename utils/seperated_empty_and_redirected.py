import os
import json
import shutil

# Define folder paths
input_folder = "combined_processed_wiki_pages"
output_folder = "redirects_or_empty"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Track files processed
total_files = 0
moved_files = 0

# Iterate over files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Ensure it's a file
    if os.path.isfile(file_path):
        total_files += 1
        try:
            # Open and parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Debug: Log content being checked
            print(f"Processing file: {filename}")
            print(f"original_text: {data.get('original_text', '')}")
            print(f"coreference_text: {data.get('coreference_text', '')}")

            # Check criteria: coreference_text is empty OR original_text starts with "REDIRECT"
            coref_empty = data.get("coreference_text", "").strip() == ""
            is_redirect = data.get("original_text", "").startswith("REDIRECT")

            if coref_empty or is_redirect:
                # Move file to the output folder
                shutil.move(file_path, os.path.join(output_folder, filename))
                moved_files += 1
                print(f"Moved: {filename}")
            else:
                # Debug: Log why the file was skipped
                print(f"Skipped: {filename} (Criteria not met)")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {filename}: Error processing file - {e}")

print(f"Processing complete. Total files: {total_files}, Files moved: {moved_files}")
