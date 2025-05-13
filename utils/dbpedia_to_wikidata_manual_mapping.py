import re

INPUT_FILE_STEP1 = "favel_test_reified_wikidata_new.ttl"
OUTPUT_FILE_FINAL = "favel_test_reified_wikidata_new_final.ttl"

# Create manual mapping for the IRI that are saved in unmapped file
MANUAL_MAPPINGS_FILE = "manual_mappings.txt"


def load_manual_mappings(filename):
    mapping_dict = {}
    with open(filename, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                dbpedia_uri, wikidata_uri = parts
                mapping_dict[dbpedia_uri] = wikidata_uri
    return mapping_dict

# Finds any dbpedia in the line that appears in manual_map,
# and replaces it with manual_map[dbpedia_uri]
def apply_manual_mappings(line: str, manual_map: dict) -> str:
    line = line.strip()
    if not line or line.startswith("#"):
        return line

    if not line.endswith(" ."):
        return line

    triple_part = line[:-2].strip()
    triple_regex = re.compile(r'^(<[^>]+>|".*?")\s+(<[^>]+>|".*?")\s+(<[^>]+>|".*?")$')
    match = triple_regex.match(triple_part)
    if not match:
        return line

    subject_str, predicate_str, object_str = match.groups()

    subject_str = replace_using_map(subject_str, manual_map)
    predicate_str = replace_using_map(predicate_str, manual_map)
    object_str = replace_using_map(object_str, manual_map)

    return f"{subject_str} {predicate_str} {object_str} ."

# If the token is an angle-bracketed DBpedia IRI that exists in manual_map
# then replace it with the mapped Wikidata IRI.
def replace_using_map(token: str, manual_map: dict) -> str:
    if token.startswith("<") and token.endswith(">"):
        inner_uri = token[1:-1]
        if inner_uri in manual_map:
            mapped_iri = manual_map[inner_uri]
            return f"<{mapped_iri}>"
    return token

# Checks if a block (subject, predicate, object, truthy value) contains any DBpedia IRI
def contains_dbpedia_iri(block: str) -> bool:
    dbpedia_regex = re.compile(r'<http://dbpedia\.org/resource/[^>]+>')
    return bool(dbpedia_regex.search(block))

def process_block(lines, manual_map):
    updated_lines = []
    for line in lines:
        updated_line = apply_manual_mappings(line, manual_map)
        updated_lines.append(updated_line)
    block_content = " ".join(updated_lines)
    if contains_dbpedia_iri(block_content):
        return None  # Skip the entire block if any DBpedia IRI remains
    return updated_lines

def main():
    manual_map = load_manual_mappings(MANUAL_MAPPINGS_FILE)
    print(f"Loaded {len(manual_map)} manual mappings.")

    with open(INPUT_FILE_STEP1, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE_FINAL, "w", encoding="utf-8") as fout:
        block = []
        for line in fin:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if stripped_line.startswith("#"):
                if block:  # Process the previous block
                    processed_block = process_block(block, manual_map)
                    if processed_block:
                        fout.writelines([l + "\n" for l in processed_block])
                    block = []
                fout.write(line)  # Write comments directly
                continue

            block.append(line)

            # Check if the line ends the block (truthy statement line assumed)
            if stripped_line.endswith("."):
                processed_block = process_block(block, manual_map)
                if processed_block:
                    fout.writelines([l + "\n" for l in processed_block])
                block = []

        # Process the last block if any
        if block:
            processed_block = process_block(block, manual_map)
            if processed_block:
                fout.writelines([l + "\n" for l in processed_block])

    print("Processing completeweeee!")

if __name__ == "__main__":
    main()
