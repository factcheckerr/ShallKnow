import requests
import os
import json
import nltk
import ast
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def load_coreference_text(processed_wiki_pages, file_name):
    """
    Loads coreference-resolved Wikipedia text from a .jsonl file and tokenizes it into sentences.

    Yields:
        Tuple[str, int]: Each sentence and its associated article ID
    """
    file_path = os.path.join(processed_wiki_pages, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            coreference_paragraph = data["coreference_text"]
            sentences = sent_tokenize(coreference_paragraph)
            data_id = data["id"]
            for sentence in sentences:
                yield sentence, data_id


def extract_triples(sentence, data_id, error_file):
    """
    Sends a sentence to the local triple extraction API and returns the response.

    Args:
        sentence (str): Sentence to extract triples from
        data_id (int): ID of the document for logging
        error_file (str): File path to log errors

    Returns:
        list or None: Extracted triples (JSON-decoded) or None if error
    """
    if sentence:
        endpoint = "http://localhost:5000/extract"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "query": sentence,
            "components": "triple_extraction"
        }
        try:
            response = requests.post(endpoint, headers=headers, data=data)
            response.raise_for_status()
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    with open(error_file, 'a') as err_f:
                        err_f.write(f"JSONDecodeError for sentence: {sentence}, ID: {data_id}\n")
                    return None
            else:
                with open(error_file, 'a') as err_f:
                    err_f.write(f"Non-200 status code for sentence: {sentence}, ID: {data_id}, Status code: {response.status_code}\n")
                return None
        except requests.exceptions.RequestException as e:
            with open(error_file, 'a') as err_f:
                err_f.write(f"RequestException for sentence: {sentence}, ID: {data_id}, Error: {str(e)}\n")
            return None

def process_files(processed_wiki_pages, output_file, error_file, processed_sentences_file):
    """
    Processes all .jsonl files containing Wikipedia articles:
    - Tokenizes coreference-resolved text into sentences
    - Sends unprocessed sentences to triple extractor API
    - Stores results in a JSONL output
    - Tracks processed sentence identifiers

    Args:
        processed_wiki_pages (str): Folder containing .jsonl files
        output_file (str): Path to save extracted triples
        error_file (str): Path to save any error logs
        processed_sentences_file (str): Path to store sentence IDs already processed
    """
    processed_sentences = set()
    if os.path.exists(processed_sentences_file):
        with open(processed_sentences_file, 'r') as f:
            processed_sentences = set(f.read().splitlines())

    if os.path.exists(output_file):
        with open(output_file, 'r') as out_f:
            for line in out_f:
                data = json.loads(line.strip())
                for triple_data in data['triples']:))
                    try:
                        if isinstance(triple_data, str):
                            triple_data_dict = ast.literal_eval(triple_data)
                        elif isinstance(triple_data, dict):
                            triple_data_dict = triple_data
                        else:
                            print(f"Unexpected triple_data type: {type(triple_data)}")
                            continue

                        claim = triple_data_dict.get("claim")
                        if claim:
                            sentence_id = f"{data['id']}_{claim}"
                            processed_sentences.add(sentence_id)

                    except (SyntaxError, ValueError) as e:
                        print(f"Error evaluating triple_data: {e}, content: {triple_data}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")

    all_files = os.listdir(processed_wiki_pages)
    list_sentence_id = set()
    num_sentences = 0
    with open(output_file, 'a') as out_f:
        for file_name in all_files:
            if file_name.endswith(".jsonl"):
                print(f"Processing file: {file_name} and no of sentences in prev. file: {num_sentences}")
                for sentence, data_id in load_coreference_text(processed_wiki_pages, file_name):
                    sentence_id = f"{data_id}_{sentence}"
                    if sentence_id in processed_sentences:
                        continue
                    result = extract_triples(sentence, data_id, error_file)

                    if result is not None:
                        output_data = {
                            "id": data_id,
                            "sentence": sentence,
                            "triples": result
                        }
                        out_f.write(json.dumps(output_data) + "\n")

                        processed_sentences.add(sentence_id)
                        list_sentence_id.add(sentence_id)

                if len(list_sentence_id)>0:
                    num_sentences = len(list_sentence_id)
                    with open(processed_sentences_file, 'a') as p_f:
                        for s_id in list_sentence_id:
                            p_f.write(f"{s_id}\n")
                    list_sentence_id.clear()
                else:
                    num_sentences = 0

if __name__ == '__main__':
    processed_wiki_pages = "wikipedia_processed_bpdp" # Input directory
    output_file = "output_triples.jsonl"
    error_file = "error_log_triple_ext.txt"
    processed_sentences_file = "filtered_processed_sentences.txt"
    process_files(processed_wiki_pages, output_file, error_file, processed_sentences_file)
