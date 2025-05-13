import json

unique_ids = set()

with open('output_triples.jsonl') as file:
    for line in file:
        item = json.loads(line.strip())  
        unique_ids.add(item['id'])      

print(len(unique_ids))  
