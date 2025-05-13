import json
import os
import re
import requests
import logging
import html
import spacy
import hashlib
from maverick import Maverick
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
import sys

from llm_query_final import LLMQuery


logging.basicConfig(filename='wikipedia_processor.log', level=logging.ERROR, format='%(asctime)s %(message)s')

input_file = './reified/BPDP_train_reified.nt'
output_dir = 'wikipedia_processed_bpdp'
os.makedirs(output_dir, exist_ok=True)

processed_titles_file = 'processed_titles_bpdp.txt'
processed_files_file = 'processed_files_bpdp.txt'
processed_hashes_file = 'processed_hashes_bpdp.txt'
error_log_file = 'content_fetch_error_titles_bpdp.txt'
last_id_file = 'last_id_bpdp.txt'
m_coref_model = Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", device="cuda:0")


def resolve_coreferences(text, title, use_maverick):
    """
    Resolves coreferences in the given text using Maverick or spaCy.

    Args:
        text (str): The input text.
        title (str): Title used for logging.
        use_maverick (bool): If True, uses Maverick; else uses spaCy.

    Returns:
        Resolved text or None if resolution fails.
    """
    try:
        if use_maverick:
            # coref_model = Maverick(hf_name_or_path="sapienzanlp/maverick-mes-ontonotes", device="cuda:0")
            result = m_coref_model.predict(text)
            return result
        else:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("fastcoref")
            doc = nlp(text)
            return doc._.resolved_text
    except Exception as e:
        # print(f"Coreference resolution failed for {title}: {e}")
        with open(error_log_file, 'a', encoding="utf-8") as f:
            f.write(title + "\n")
        return None


def generate_resolved_text(model_output):
    """
    Generates a resolved text version by replacing mentions with their antecedents
    based on Maverick's output clusters.

    Args:
        model_output (dict): Output from Maverick.

    Returns:
        str: Coreference-resolved version of input text.
    """
    tokens = model_output.get("tokens", [])
    resolved_tokens = tokens.copy()
    replacement_map = {}

    for cluster in model_output.get("clusters_token_offsets", []):
        if len(cluster) < 2:
            continue

        antecedent_start, antecedent_end = cluster[0]
        antecedent = " ".join(tokens[antecedent_start:antecedent_end + 1])

        for mention in cluster[1:]:
            start, end = mention
            replacement_map[(start, end)] = antecedent

    for (start, end), antecedent in sorted(replacement_map.items(), key=lambda x: x[0][0]):
        for i in range(start, end + 1):
            resolved_tokens[i] = None
        resolved_tokens[start] = antecedent

    resolved_tokens = [token for token in resolved_tokens if token is not None]
    resolved_text = " ".join(resolved_tokens)
    return resolved_text


def load_processed_titles():
    """
    Loads the set of processed (title, entity_pair) entries from file.

    Returns:
        Set of tuples: (normalized_title, entity_pair)
    """
    processed_data = set()
    if os.path.exists(processed_titles_file):
        with open(processed_titles_file, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) == 2:
                    processed_data.add((parts[0], parts[1]))
    return processed_data


def save_processed_title(title, entity_pair):
    """
    Saves a processed (title, entity_pair) entry if not already logged.
    """
    processed_titles = load_processed_titles()
    entry = (title, entity_pair)
    if entry not in processed_titles:
        with open(processed_titles_file, 'a', encoding="utf-8") as f:
            f.write(f"{title} | {entity_pair}\n")


def load_processed_files():
    """
    Loads the set of processed (filepath, entity_pair) entries.
    """
    processed_data = set()
    if os.path.exists(processed_files_file):
        with open(processed_files_file, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) == 2:
                    processed_data.add((parts[0], parts[1]))  # (file_path, entity_pair)
    return processed_data


def save_processed_file(file_path, entity_pair):
    """
    Saves a processed (file_path, entity_pair) entry.
    """
    processed_files = load_processed_files()
    entry = (file_path, entity_pair)
    if entry not in processed_files:
        with open(processed_files_file, 'a', encoding="utf-8") as f:
            f.write(f"{file_path} | {entity_pair}\n")


def load_processed_hashes():
    """Load content hashes of processed articles to avoid duplicate content."""
    if os.path.exists(processed_hashes_file):
        with open(processed_hashes_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def save_processed_hash(hash_value):
    """Append a processed content hash to the file."""
    with open(processed_hashes_file, 'a') as f:
        f.write(hash_value + "\n")


def load_last_id():
    """Load the last used incremental id from file; default to 1 if not present."""
    if os.path.exists(last_id_file):
        with open(last_id_file, 'r') as f:
            try:
                return int(f.read().strip())
            except:
                return 1
    return 1


def save_last_id(last_id):
    """Persist the last used incremental id to file."""
    with open(last_id_file, 'w') as f:
        f.write(str(last_id))


def normalize_filename(filename):
    """Normalize filename by replacing underscores with spaces and converting to lowercase."""
    filename = re.sub(r'[\\/:"*?<>|]', '_', filename)  # Replaces / and other invalid filename characters
    return filename.replace("_", " ").strip().lower()


def compute_hash(text):
    """Compute an MD5 hash for the given text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_data(input_file):
    """
    Loads and parses reified RDF triples from a Turtle (.nt) file.
    Only includes triples where hasTruthValue == 1.0.

    Returns:
        List of (subject_uri, predicate_uri, object_uri)
    """
    # print("Loading RDF data")
    data = []
    g = Graph()
    g.parse(input_file, format='nt')

    truth_predicate = URIRef("http://swc2017.aksw.org/hasTruthValue")

    for stmt in g.subjects(RDF.type, RDF.Statement):
        truth_value = g.value(stmt, truth_predicate)
        if truth_value is None or str(truth_value) != "1.0":
            continue

        subj = g.value(stmt, RDF.subject)
        pred = g.value(stmt, RDF.predicate)
        obj = g.value(stmt, RDF.object)

        if subj and pred and obj:
            data.append((str(subj), str(pred), str(obj)))

    return data


def fetch_wikipedia_content(title):
    """
    Fetches plain-text Wikipedia content and outgoing links using the MediaWiki API.

    Args:
        title (str): Wikipedia article title

    Returns:
        Tuple[str, List[str]]: Page text and list of linked article titles
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|links",
        "explaintext": 1,
        "redirects": 1,
        "pllimit": "max",
        "titles": title
    }
    response = requests.get(url, params=params)
    data = response.json()
    query = data.get("query", {})
    # Handle redirects if present
    if "redirects" in query:
        redirected_title = query["redirects"][0]["to"]
        return fetch_wikipedia_content(redirected_title)

    pages = query.get("pages", {})
    for page_id, page_data in pages.items():
        if "missing" in page_data:
            return None, []
        text = page_data.get("extract", "")
        links = []
        if "links" in page_data:
            links.extend([link["title"] for link in page_data["links"]])
        # Continue fetching additional links if available
        while "continue" in data and "plcontinue" in data["continue"]:
            params["plcontinue"] = data["continue"]["plcontinue"]
            response = requests.get(url, params=params)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, p_data in pages.items():
                if "links" in p_data:
                    links.extend([link["title"] for link in p_data["links"]])
        return text, links
    return None, []


def clean_text(text):
    """
    Cleans raw Wikipedia markup and HTML from text.

    Args:
        text (str): Raw article text

    Returns:
        str: Cleaned text
    """
    # Decode Unicode escape sequences
    text = text.encode("utf-8").decode("utf-8") #decode("unicode_escape")
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove Wikipedia section headings (e.g., "== some section ==")
    text = re.sub(r'={2,}.*?={2,}', '', text)
    # Remove wiki links but keep display text (e.g., [[something|something]] → something)
    text = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', text)
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    # Remove citations and references (e.g., <ref>some text</ref>)
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove any remaining HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove templates (e.g., {{Infobox}}, {{Cite web}})
    text = re.sub(r'\{\{.*?\}\}', '', text)
    # Remove unwanted characters like extra backslashes, quotes, and artifacts
    text = text.replace("\\", "").replace("'", "").replace('"', '')
    # Remove content inside parentheses if it contains non-ASCII characters (e.g., IPA, transliterations)
    text = re.sub(r'\(\s*[^()]*[^\x00-\x7F]+[^()]*\s*\)', '', text)
    # Remove empty parentheses
    text = re.sub(r'\(\s*\)', '', text)
    # Normalize spaces but keep paragraphs intact
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Ensure a newline exists before each paragraph (to prevent accidental merging)
    text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)
    return text

def compute_coref(cleaned_text, title):
    """
    Applies Maverick-based coreference resolution with fallback handling.

    Args:
        cleaned_text (str): Text to resolve
        title (str): For logging

    Returns:
        str or False: Resolved text or False on error
    """
    # Process the main article (but postpone marking as processed)
    model_output = resolve_coreferences(cleaned_text, title, True)
    if model_output:
        if isinstance(model_output, dict) and "tokens" in model_output:
            coreference_text = generate_resolved_text(model_output)
        else:
            coreference_text = resolve_coreferences(cleaned_text, title, False)
    else:
        with open(error_log_file, 'a',encoding="utf-8") as f:
            f.write(title + "\n")
        return False

    return coreference_text


def process_triples(model:str):
    """
    Processes RDF triples, fetching and cleaning related Wikipedia articles,
    resolving coreferences, refining text via LLM, and saving outputs.

    Args:
        model (str): Name of the local LLM model to use for text refinement.
    """
    data = load_data(input_file)
    processed_titles = load_processed_titles()
    processed_files = load_processed_files()
    processed_hashes = load_processed_hashes()
    last_id = load_last_id()
    llm = LLMQuery("http://localhost:11434/api/generate")

    for subject_uri, _, object_uri in data:
        # Extract and normalize titles
        sub_title = os.path.basename(subject_uri)
        obj_title = os.path.basename(object_uri)

        sub_normalized_title = normalize_filename(sub_title)
        obj_normalized_title = normalize_filename(obj_title)
        entity_pair = f"{sub_normalized_title}_{obj_normalized_title}"

        print(f"{sub_normalized_title}---{obj_normalized_title}")

        for uri in [subject_uri, object_uri]:
            title = os.path.basename(uri)
            normalized_title = normalize_filename(title)

            output_path = os.path.join(output_dir, f"{title}.jsonl")
            # normalized_output_path = normalize_filename(output_path)

            if (normalized_title, entity_pair) in processed_titles or (output_path, entity_pair) in processed_files:
                print(f"Skipping already processed title-file pair: {normalized_title} ({entity_pair})")
                continue

            print(f"Processing title: {title} for entity pair {entity_pair}")
            content, links = fetch_wikipedia_content(title)
            if not content:
                continue

            cleaned_text = clean_text(content)

            hash_value = compute_hash(cleaned_text)
            if hash_value in processed_hashes:
                print(f"Skipping duplicate content for: {title}")
                continue
            # cleaned_text = str(cleaned_text).replace("\n", " ")

            # cleaned_text = clean_text(content)
            coreference_text = ""

            # co_ref = compute_coref(cleaned_text, title)
            # if co_ref==False:
            #     continue
            # else:
            #     coreference_text = co_ref
            coreference_text = cleaned_text

            llm_cleaned_text = llm.get_response_from_api_call(
                cleaned_text,
                sub_normalized_title, obj_normalized_title, model=model)

            llm_coreference_text = ""
            co_ref = compute_coref(llm_cleaned_text, title)
            if co_ref == False:
                print("error in Coref file: " + title)
                continue
            else:
                llm_coreference_text = co_ref

            # Process outgoing links first
            for link in links:
                normalized_link = normalize_filename(link)
                if (normalized_link, entity_pair) in processed_titles:
                    print(f"Skipping already processed title-file pair: {normalized_title} ({entity_pair})")
                    continue

                print(f"Processing linked page: {normalized_link} for entity pair {entity_pair}")
                link_content, _ = fetch_wikipedia_content(link)
                if not link_content:
                    continue

                cleaned_link_text = clean_text(link_content)

                link_hash = compute_hash(cleaned_link_text)
                if link_hash in processed_hashes:
                    print(f"Skipping duplicate content for link: {link}")
                    continue

                # col_ref = compute_coref(cleaned_link_text, title)
                # if col_ref == False:
                #     continue
                # else:
                #     link_coref_text = col_ref
                link_coref_text = cleaned_link_text
                # LLM
                link_coref_text_llm = llm.get_response_from_api_call(
                        cleaned_link_text,
                        sub_normalized_title, obj_normalized_title, model=model)

                col_ref = compute_coref(link_coref_text_llm, title)
                if col_ref == False:
                    print("error in Coref file: "+ title)
                    continue
                else:
                    link_coref_text_llm = col_ref

                if link_coref_text and "API error" not in link_coref_text_llm:
                    link_output_path = os.path.join(output_dir, normalize_filename(f"{link}.jsonl"))
                    with open(link_output_path, 'a') as f:
                        json.dump({"id": last_id, "original_text": cleaned_link_text, "coref_text_llm": link_coref_text_llm}, f) #, "coref_text": link_coref_text
                        f.write("\n")
                    save_processed_title(link, entity_pair)
                    save_processed_file(link_output_path, entity_pair)
                    processed_hashes.add(link_hash)
                    save_processed_hash(link_hash)
                    last_id += 1
                    save_last_id(last_id)

            if coreference_text and "API error" not in llm_coreference_text:
                # Now save the main article after processing its links
                with open(output_path, 'a') as f:
                    json.dump({"id": last_id, "original_text": cleaned_text, "coref_text_llm":llm_coreference_text}, f) #, "coref_text": coreference_text
                    f.write("\n")
                save_processed_title(title, entity_pair)
                save_processed_file(output_path, entity_pair)
                processed_hashes.add(hash_value)
                save_processed_hash(hash_value)
            last_id += 1
            save_last_id(last_id)

if __name__ == '__main__':
#    reprocess_files()
#    exit(1)
    args = sys.argv[1:]  # Exclude script name
    modl = args[0] # LLM model name passed from command-line
    process_triples(model=modl)
