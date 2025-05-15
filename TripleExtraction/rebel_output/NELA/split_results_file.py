import json
import os

def split_jsonl(input_file, lines_per_file):
    # Create a directory to store split files
    os.makedirs("split_files", exist_ok=True)

    with open(input_file, 'r') as infile:
        file_count = 0
        line_count = 0
        current_output_file = None

        for line in infile:
            if line_count % lines_per_file == 0:
                if current_output_file:
                    current_output_file.close()
                file_count += 1
                current_output_file = open(f"split_files/output_{file_count}.jsonl", 'w')

            current_output_file.write(line)
            line_count += 1

        if current_output_file:
            current_output_file.close()

# Example usage:
input_jsonl_file = 'nela_10_train_coref.jsonl'  # Replace with your JSONL file path
lines_per_output_file = 10000  # Specify the number of lines per output file

split_jsonl(input_jsonl_file, lines_per_output_file)
