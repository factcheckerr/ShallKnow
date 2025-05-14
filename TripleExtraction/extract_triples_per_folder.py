import logging

import requests
import os
import json
import nltk
import ast
import re
import hashlib
import html

from nltk.tokenize import sent_tokenize
import threading
# Create a global lock for file writing
file_write_lock = threading.Lock()

nltk.download('punkt')
nltk.download('punkt_tab')

WIKI_SECTION_RE = re.compile(r'={2,}.*?={2,}')
WIKI_LINK_DISPLAY_RE = re.compile(r'\[\[.*?\|(.*?)\]\]')
WIKI_LINK_RE = re.compile(r'\[\[(.*?)\]\]')
REF_TAG_RE = re.compile(r'<ref.*?>.*?</ref>', flags=re.DOTALL)
HTML_TAG_RE = re.compile(r'<.*?>')
TEMPLATE_RE = re.compile(r'\{\{.*?\}\}')
PAREN_NON_ASCII_RE = re.compile(r'\(\s*[^()]*[^\x00-\x7F]+[^()]*\s*\)')
EMPTY_PAREN_RE = re.compile(r'\(\s*\)')

SPACY_MODEL = None

text_tag = os.environ.get('TEXT_TAG', default='coref_text_llm')



def contains_invalid_phrase(sentence: str) -> bool:
    invalid_keywords = [
        "the content does not",
        "no direct or indirect mention",
        "The provided content does not",
        "are not mentioned or connected in the provided content",
        "do not contain any direct or indirect mentions",
        "does not mention",
        "no information",
        "not established",
        "cannot be established",
        "does not contain",
        "is missing",
        "not related to",
        "does not explicitly mention",
        "there is no mention",
        "does not include any information",
        "no direct or indirect connection mentioned",
        "does not explicitly connect",
        "does not explicitly detail"
    ]
    sentence_lower = sentence.lower()
    return any(keyword in sentence_lower for keyword in invalid_keywords)

def load_coreference_text(processed_wiki_pages, file_name):
    file_path = os.path.join(processed_wiki_pages, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            coreference_paragraph = data["coreference_text"]
            sentences = sent_tokenize(coreference_paragraph)
            data_id = data["id"]
            for sentence in sentences:
                yield sentence, data_id


def load_coreference_paragrah(processed_wiki_pages, file_name):
    file_path = os.path.join(processed_wiki_pages, file_name)
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            coreference_paragraph = str(data[text_tag]).replace("LLM Output :","")
            if "no response" not in str(coreference_paragraph).lower():
                # sentences = sent_tokenize(coreference_paragraph)
                data_id = data["id"]
                yield coreference_paragraph, data_id


def remove_special_characters(paragraph):
    cleaned_paragraph = re.sub(r'[^a-zA-Z0-9\s.]', '', paragraph)
    return cleaned_paragraph

def extract_triples(paragraph, data_id, error_file):
    if paragraph:
        cleaned_paragraph = remove_special_characters(paragraph)
        # query_paragraph = json.dump(paragraph)
        endpoint = "http://localhost:5000/dextract"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "query": cleaned_paragraph,
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
                        err_f.write(f"JSONDecodeError for ID: {data_id}\n")
                    return None
            else:
                with open(error_file, 'a') as err_f:
                    err_f.write(f"Non-200 status code for ID: {data_id}, Status code: {response.status_code}\n")
                return None
        except requests.exceptions.RequestException as e:
            with open(error_file, 'a') as err_f:
                err_f.write(f"RequestException for ID: {data_id}, Error: {str(e)}\n")
            return None

def process_files(processed_wiki_pages, output_file, error_file, processed_sentences_file):
    processed_sentences = set()
    if os.path.exists(processed_sentences_file):
        with open(processed_sentences_file, 'r') as f:
            processed_sentences = set(f.read().splitlines())

    if os.path.exists(output_file):
        with open(output_file, 'r') as out_f:
            for line in out_f:
                data = json.loads(line.strip())
                sentence_id = f"{data['id']}_{data['sentence']}"
                processed_sentences.add(sentence_id)

    all_files = os.listdir(processed_wiki_pages)
    list_sentence_id = set()
    num_sentences = 0
    with open(output_file, 'a') as out_f:
        for file_name in all_files:
            if file_name.endswith(".jsonl"):
                print(f"Processing file: {file_name} and no of sentences in prev. file: {num_sentences}")
                for sentence, data_id in load_coreference_text(processed_wiki_pages, file_name):
                    sentence_id = f"{data_id}_{sentence}"
                    # list_sentence_id.add(sentence_id)

                    if sentence_id in processed_sentences:
                        continue

                    # print("###################")
                    # print(sentence, data_id)
                    result = extract_triples(sentence, data_id, error_file)

                    if result is not None:
                        output_data = {
                            "id": data_id,
                            "sentence": sentence,
                            "triples": result
                        }
                        out_f.write(json.dumps(output_data) + "\n")
                        cleaned_text=clean_text(sentence_id)
                        processed_sentences.add(cleaned_text)
                        list_sentence_id.add(cleaned_text)

                if len(list_sentence_id)>0:
                    num_sentences = len(list_sentence_id)
                    with open(processed_sentences_file, 'a') as p_f:
                        for s_id in list_sentence_id:
                            p_f.write(f"{s_id}\n")
                    list_sentence_id.clear()
                else:
                    num_sentences = 0
global num_sentences
num_sentences = 0
global processed_sentences
processed_sentences = set()
global processed_hashes
processed_hashes = set()
processed_hashes_lock = threading.Lock()
def does_not_start_with_number(s):
    return not s[0].isdigit() if s else True  # Return True for empty string
def read_files_in_folder(path, processed_wiki_pages):

    print("reading ")

    global processed_sentences, processed_hashes
    processed_hashes = set()
    processed_sentences = set()

    processed_wiki_pages = os.path.join(path, processed_wiki_pages)
    output_file = os.path.join(path, "output_triples.jsonl")
    error_file = os.path.join(path, "error_log_triple_ext.txt")
    processed_sentences_file = os.path.join(path, "filtered_processed_sentences.txt")
    processed_articles_hashes = os.path.join(path, "processed_articles_hashes.txt")


    # Load previously processed hashes
    processed_hashes = load_processed_hashes(processed_articles_hashes)

    # process_files_per_article(processed_wiki_pages,output_file,error_file,processed_sentences_file)

    # Load processed sentences from file
    if os.path.exists(processed_sentences_file):
        with open(processed_sentences_file, 'r') as f:
            for line in f.read().splitlines():
                if line:
                    cleaned_text = clean_text(line)
                    processed_sentences.add(cleaned_text)
                    link_hash = compute_hash(cleaned_text)
                    processed_hashes.add(link_hash)

    if os.path.exists(output_file):
        with open(output_file, 'r') as out_f:
            for line in out_f:
                if line.strip():
                    data = json.loads(line.strip())
                    sentence_id = f"{data['id']}_{data['sentence']}"
                    print("already processed: "+str(data['id']))
                    cleaned_text = clean_text(sentence_id)
                    processed_sentences.add(cleaned_text)
                    processed_hashes.add(compute_hash(cleaned_text))

    new_keys = processed_sentences
    with open(processed_sentences_file, 'w') as p_f:
        for key in new_keys:
            p_f.write(f"{key}\n")

    print("Total processed sentences:", len(processed_sentences))

    all_files = os.listdir(processed_wiki_pages)
    paragraphs = []

    for file_name in all_files:
        if not file_name.endswith(".jsonl"):
            continue
        try:
            print(f"Scanning file: {file_name}, sentences so far: {len(processed_sentences)}")
            for  article_paragraph, data_id in load_coreference_paragrah(processed_wiki_pages, file_name):
                skip = False
                for sent in [article_paragraph]:
                    cleaned_text = clean_text(f"{data_id}_{sent}")
                    if data_id==5755:
                        print(f"{data_id} and {sent}: {cleaned_text}")
                    if cleaned_text in processed_sentences:
                        print(f"Skipping already processed: {cleaned_text}")
                        skip = True
                        break
                    else:
                        print("to process: ", cleaned_text)
                if not skip:
                    if not contains_invalid_phrase(article_paragraph):
                        paragraphs.append({data_id: article_paragraph})
                    else:
                        print(f"Skipping invalid article: {article_paragraph}")

        except Exception as e:
            with open(error_file, 'a') as err_f:
                err_f.write(f"Error processing {file_name}: {str(e)}\n")
    print("Remaining files to process: ", len(paragraphs))
    return paragraphs

def store_results(output_file, result, data_id, processed_sentences_file,processed_hashes_file_path):
    list_sentence_id = set()
    if result is None:
        return
    with file_write_lock:  # Lock ensures exclusive access during file writing
        with open(output_file, 'a') as out_f:
            for res in result:
                # res = json.loads(res)
                claim = res["claim"].replace("LLM output:-", "").strip()
                if not claim:
                    continue
                output_data = {
                    "id": data_id,
                    "sentence": claim,
                    "triples": res
                }
                out_f.write(json.dumps(output_data) + "\n")

                sentence_id = f"{data_id}_{claim}"
                cleaned_text = clean_text(sentence_id)

                if cleaned_text not in processed_sentences:
                    processed_sentences.add(cleaned_text)
                    list_sentence_id.add(cleaned_text)

                link_hash = compute_hash(cleaned_text)
                if link_hash not in processed_hashes:
                    save_processed_hash(link_hash, processed_hashes_file_path)
                    processed_hashes.add(link_hash)

    if list_sentence_id:
        with file_write_lock:  # Lock ensures exclusive access during file writing
            with open(processed_sentences_file, 'a') as p_f:
                for s_id in list_sentence_id:
                    p_f.write(f"{s_id}\n")
        list_sentence_id.clear()

    print(f"Number of sentences saved: {len(list_sentence_id)}")


def process_files_per_article(processed_wiki_pages, output_file, error_file, processed_sentences_file):
    processed_sentences = set()
    if os.path.exists(processed_sentences_file):
        with open(processed_sentences_file, 'r') as f:
            processed_sentences = set(f.read().splitlines())

    if os.path.exists(output_file):
        with open(output_file, 'r') as out_f:
            for line in out_f:
                data = json.loads(line.strip())
                for triple_data in data['triples']:
                    # print(triple_data)
                    # print(type(triple_data))
                    try:
                        if isinstance(triple_data, str):
                            triple_data_dict = ast.literal_eval(triple_data)
                        elif isinstance(triple_data, dict):
                            triple_data_dict = triple_data
                        else:
                            print(f"Unexpected triple_data type: {type(triple_data)}")
                            continue
                        # print(type(triple_data_dict))

                        claim = triple_data_dict.get("claim")
                        if claim:
                            sentence_id = f"{data['id']}_{claim}"
                            processed_sentences.add(sentence_id)
                            # print(sentence_id)

                    except (SyntaxError, ValueError) as e:
                        print(f"Error evaluating triple_data: {e}, content: {triple_data}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")

    all_files = os.listdir(processed_wiki_pages)
    list_sentence_id = set()
    num_sentences = 0
    with open(output_file, 'a') as out_f:
        for file_name in all_files:
            # input_file_path = os.path.join(processed_wiki_pages, file_name)
            if  not file_name.endswith(".jsonl"):
                continue
            try:
                print(f"Processing file: {file_name} and no of sentences in prev. file: {num_sentences}")
                for sentences, article_paragraph, data_id in load_coreference_paragrah(processed_wiki_pages, file_name):
                    # sentence_id = f"{data_id}_{sentence}"
                    first_sentence_id = f"{data_id}_{sentences[0]}"
                    # list_sentence_id.add(sentence_id)

                    # if sentence_id in processed_sentences:
                    #     continue

                    if first_sentence_id in processed_sentences:
                        continue

                    result = extract_triples(article_paragraph, data_id, error_file)
                    if result is not None:
                        output_data = {
                            "id": data_id,
                            "triples": result
                        }
                        out_f.write(json.dumps(output_data) + "\n")

                        for sentence in sentences:
                            sentence_id = f"{data_id}_{sentence}"
                            sentence_id = clean_text(sentence_id)
                            processed_sentences.add(sentence_id)
                            list_sentence_id.add(sentence_id)

                if len(list_sentence_id) > 0:
                    num_sentences = len(list_sentence_id)
                    with open(processed_sentences_file, 'a') as p_f:
                        for s_id in list_sentence_id:
                            p_f.write(f"{s_id}\n")
                    list_sentence_id.clear()
                else:
                    num_sentences = 0

            except Exception as e:
                with open(error_file, 'a') as err_f:
                    err_f.write(f"Error processing {file_name}: {str(e)}\n")


def load_processed_hashes(processed_hashes_file):
    if os.path.exists(processed_hashes_file):
        with open(processed_hashes_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()


def save_processed_hash(hash_value, processed_hashes_file):
    with processed_hashes_lock:
        with open(processed_hashes_file, 'a') as f:
            f.write(hash_value + "\n")

def compute_hash(text):
    """Compute an MD5 hash for the given text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def clean_text(text):
    text = text.encode("utf-8").decode("utf-8")
    text = html.unescape(text)
    text = WIKI_SECTION_RE.sub('', text)
    text = WIKI_LINK_DISPLAY_RE.sub(r'\1', text)
    text = WIKI_LINK_RE.sub(r'\1', text)
    text = REF_TAG_RE.sub('', text)
    text = HTML_TAG_RE.sub('', text)
    text = TEMPLATE_RE.sub('', text)
    text = text.replace("\\", "").replace("'", "").replace('"', '')
    text = PAREN_NON_ASCII_RE.sub('', text)
    text = EMPTY_PAREN_RE.sub('', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)
    return text

if __name__ == '__main__':
    processed_wiki_pages = "/data/nebula/WikipediaExtractor/"
    folder = "wikipedia_processed_test4"
    output_file = "output_triples.jsonl"
    error_file = "error_log_triple_ext.txt"
    processed_sentences_file = "filtered_processed_sentences.txt"
    processed_articles_file = "processed_articles.txt"

    # process_files(processed_wiki_pages, output_file, error_file, processed_sentences_file)
    # process_files_per_article(processed_wiki_pages, output_file, error_file, processed_sentences_file)
    read_files_in_folder(processed_wiki_pages, folder)