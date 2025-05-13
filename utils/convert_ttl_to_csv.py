import re
import csv
from collections import defaultdict

# Input and output file paths
input_file = 'BPDP_train_reified_wikidata_final.ttl'
output_file = 'bpdp_train.csv'

# Dictionary to store statement triples
statements = defaultdict(dict)

# Regular expression to parse each triple
pattern = re.compile(r'<([^>]+)> <([^>]+)> (?:"([^"]+)"\^\^<[^>]+>|<([^>]+)>) \.')

# Read and parse the file
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = pattern.match(line.strip())
        if match:
            subj, pred, literal_obj, uri_obj = match.groups()
            obj = literal_obj if literal_obj else uri_obj
            stmt_id = subj
            if pred.endswith('hasTruthValue'):
                # Convert score to int: "1.0" -> 1, "0.0" -> 0
                statements[stmt_id]['score'] = int(float(obj))
            elif pred.endswith('#subject'):
                statements[stmt_id]['subject'] = obj
            elif pred.endswith('#predicate'):
                statements[stmt_id]['predicate'] = obj
            elif pred.endswith('#object'):
                statements[stmt_id]['object'] = obj

# Write to CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['subject', 'predicate', 'object', 'score'])  # Header
    for stmt in statements.values():
        if all(key in stmt for key in ('subject', 'predicate', 'object', 'score')):
            writer.writerow([stmt['subject'], stmt['predicate'], stmt['object'], stmt['score']])

print(f"Done!")
