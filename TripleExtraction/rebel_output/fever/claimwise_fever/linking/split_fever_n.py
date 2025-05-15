import os

input_file = 'combined_fever_data.jsonl'
output_folder = "output_triples_split"
os.makedirs(output_folder, exist_ok=True)
num_splits = 10
lines = set()
with open(input_file, 'r') as infile:
    lines = infile.readlines()
lines_per_split = len(lines) // num_splits
remainder = len(lines) % num_splits
start = 0
for i in range(1, num_splits + 1):
    end = start + lines_per_split + (1 if i <= remainder else 0)
    split_lines = lines[start:end]
    output_file = os.path.join(output_folder, f"output_triples_train_{i}.jsonl")
    with open(output_file, 'a') as outfile:
        outfile.writelines(split_lines)
    start = end

print(f"Split complete!! Files saved in '{output_folder}'")
