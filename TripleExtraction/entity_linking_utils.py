# parser.add_argument("--path",  default='/upb/users/u/uqudus/profiles/unix/cs/NEBULA/TripleExtraction/rebel_output/FEVER/output/output_train/')
import json
import logging
import os
import requests
import jsonlines
import nltk
# from threaded_extraction import check_if_synonyms_in_sentence

import itertools
# nltk.download(download_dir='/data/jetbrains/treebank')

import configparser

from datetime import datetime
from typing import Set
from nltk.tokenize import word_tokenize
import unicodedata
from LLM_query import is_contextual_match
from nltk.corpus import wordnet
import threading
from rapidfuzz import fuzz
import spacy
import logging
import re
# Create a global lock for file writing
file_write_lock = threading.Lock()

# logging.basicConfig(level=logging.INFO)
dataset_path = ""
config = configparser.ConfigParser()
config_file = 'configuration.ini'
config.read(config_file)
log_path = config.get('logging', 'logs_path', fallback='nebula.log')
log_level = os.environ.get('log_level', default='ERROR') #config.get('logging', 'log_level', fallback='INFO')
log_level = getattr(logging, log_level.upper(), logging.INFO)
logging.basicConfig(filename=log_path, level=log_level,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='w')
nltk_download_path = config.get('logging', 'nltk_download_path')


nltk.download('treebank', download_dir=nltk_download_path)
nltk.download('punkt',download_dir=nltk_download_path)
nltk.download('wordnet',download_dir=nltk_download_path)
# nltk.download('punkt')

nltk.download('omw-1.4',download_dir=nltk_download_path)  # Optional: additional multilingual WordNet data
nltk.download('averaged_perceptron_tagger',download_dir=nltk_download_path)  # For POS tagging, if needed
nltk.download('punkt',download_dir=nltk_download_path)  # For tokenization, if needed

global  relations_dict
relations_dict = dict()

global  entities_dict
entities_dict = dict()

global  error_ent_dict
error_ent_dict = dict()

nltk.data.path.append(nltk_download_path)
processed_claims = []
class ArgumentParserMock:
    def __init__(self):
        self.args = {}

    def add_argument(self, name, default=None):
        self.args[name] = default

    def parse_args(self, args=[]):
        return self.args


        # 'http://porque.cs.upb.de/porque-neamt/custom-pipeline'
        # --header 'Content-Type: application/x-www-form-urlencoded'
        # --data-urlencode 'components=babelscape_ner, mgenre_el'
        # --data-urlencode 'query=A message from the Pittsburgh School of Medicine, where he worked as an assistant professor, says Liu was conducting “significant” research into the coronavirus.'
        # --data-urlencode 'target_lang=en'

def get_synonyms(word):
    # nltk.download('wordnet', download_dir=nltk_download_path)
    synonyms = set()
    synnt = list()
    try:
        words = word.split(' ')
        for w in words:
            synnt += wordnet.synsets(w)
    except Exception as e:
        print(f'caught {type(e)}: e')
        exit(1)
    for synset in synnt:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms


import re
import unicodedata
from typing import Set


def get_all_representations(word: str) -> Set[str]:
    """Generate all possible Unicode representations of a word"""
    representations = set()
    word = word.lower().strip()

    # Add normalized forms
    representations.add(unicodedata.normalize('NFC', word))
    representations.add(unicodedata.normalize('NFD', word))

    # Add escaped versions
    try:
        escaped = word.encode('unicode-escape').decode('ascii')
        representations.add(escaped)
        representations.add(escaped.replace('\\', '\\\\'))  # Double-escaped
    except:
        pass

    return representations


def check_if_synonyms_in_sentence(sentence: str, word: str) -> bool:
    """
    Complete solution that handles:
    - Special characters (ü, \u00fc, \\u00fc)
    - Synonym matching
    - Contextual fallback
    """
    # Get all representations of the target word
    target_forms = get_all_representations(word)

    # Get synonyms and their representations
    try:
        synonyms = get_synonyms(word)
        for syn in synonyms:
            target_forms.update(get_all_representations(syn))
    except Exception as e:
        logging.error(f"Synonym error: {e}")

    # Check all representations against the sentence
    sentence_lower = sentence.lower()
    for form in target_forms:
        # Direct match
        if form in sentence_lower:
            return True
        # Word-boundary match
        if re.search(rf'(^|\W){re.escape(form)}($|\W)', sentence_lower):
            return True

    # Final fallback: contextual matching
    return is_contextual_match(
        word,
        [t for t in re.findall(r'\w+', sentence) if t.isalpha()],
        similarity_threshold=0.7
    )

# def check_if_synonyms_in_sentence(sentence, word):
#     # logging.info("checking syn"+str(word)+ "---"+str(sentence))
#     try:
#         synonyms = get_synonyms(word)
#     except Exception as e:
#         logging.error("Problem with error: {e}")
#         synonyms = []
#
#     # print("word:"+word)
#     # if(len(synonyms)>0):
#     #     print("ssynonyms:"+str(synonyms))
#     synonyms.add(word)
#     try:
#         words_in_sentence = set(word_tokenize(sentence.lower()))
#     except Exception as e:
#         print(f'caught {type(e)}: e')
#         exit(1)
#
#     # print(len(synonyms))
#     missing_synonyms = [synonym for synonym in synonyms if synonym not in words_in_sentence]
#     # print(len(missing_synonyms))
#     if len(synonyms) > len(missing_synonyms):
#         return True
#     else:
#         if is_contextual_match(word, sentence.split(' '),similarity_threshold=0.7):
#             return True
#         return False
def restrict_sentence_length(sentence, max_length=300):
    # Trim the sentence to the maximum length if it exceeds 300 characters
    if len(sentence) > max_length:
        sentence = sentence[:max_length].rstrip()  # remove any trailing spaces
    return sentence
def get_triples(query):
    headers = {
        'accept': 'application/json',
    }

    params = {
        'text': restrict_sentence_length(query),
        'is_split_into_words': 'false',
        'retriever_batch_size': '32',
        'reader_batch_size': '32',
        'return_windows': 'false',
        'use_doc_topic': 'false',
        'annotation_type': 'char',
        'passage_max_length': '1000',
        'relation_threshold': '0.5',
    }

    try:
        # Make the POST request to the NER/EL pipeline
        response = requests.get(os.environ.get('RELIK_URL', default='http://127.0.0.1:12346/api/relik'), params=params, headers=headers)
        # response = requests.get(os.environ.get('RELIK_URL', default='http://localhost:12345/api/relik'), params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad HTTP responses (4xx, 5xx)
        logging.info("length:" + str(len(query)))
    except requests.exceptions.RequestException as e:
        # Handle network errors or bad responses
        logging.error("This query failed:"+str(query))
        logging.error(f"RELIK failed: Exception: {e}")
        return []
    link_info = response.json()
    triplets = []
    if response.status_code == 200:
        if isinstance(link_info,list):
            link_info = link_info[0]
        # print(link_info)
        test_relations =link_info['candidates']['triplet'][0][0]
        if isinstance(test_relations,list):
            logging.info("we can extract relations from here")
            for rel in test_relations:
                if rel['text'] not in relations_dict:
                    relations_dict[rel['text']]= 'http://www.wikidata.org/entity/'+rel['metadata']['property']
                    relations_cache_updated = True
        if 'triplets' not in link_info.keys():
            return triplets
        for ent in link_info['triplets']:
            if len(ent)!=4:
                logging.info('entitty miss match')
            else:
                triplets.append({'head': ent[0][3], 'type': ent[1], 'tail': ent[2][3]})
    elif response.status_code == 500:
        logging.error(link_info)
        logging.error("service internal error occurs: http://127.0.0.1:12345/api/relik")
    else:
        logging.error(str(response.status_code))
        logging.error(link_info)
        logging.error("service unresponsive: http://127.0.0.1:12345/api/relik")
        exit(1)
    return triplets


def dump_malformed_json(json_str, filename='malformed_json.txt'):
    """
    Writes the malformed JSON string to a file.

    Args:
        json_str (str): The JSON string to be written.
        filename (str): The file where the JSON is dumped (default is 'malformed_json.txt').
    """
    try:
        # Ensure the directory exists (optional, in case of specific paths)# Get the current working directory
        # current_directory = os.getcwd()# Get the current working directory
        current_directory = os.getcwd()
        os.makedirs(os.path.dirname(current_directory+"/"+filename), exist_ok=True)# Timestamp the log entry for tracking purposes
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "response_content": json_str
        }

        # Append the malformed JSON to a file
        with open(current_directory+"/"+filename, 'a') as file:
            file.write(json_str +"--"+ str(log_entry) +"\n")

        logging.info(f"Malformed JSON successfully dumped to {filename}.")

    except Exception as e:
        logging.error(f"Failed to dump malformed JSON: {e}")



def ner_el_func(query, entities):
    """
    Sends a query to a custom Named Entity Recognition (NER) and Entity Linking (EL) pipeline and returns the recognized entities.

    Args:
        query (str): The input query text.

    Returns:
        list: A list of recognized entities from the query. flair_ner, davlan_ner or spacy_ner.
    """
    components = "spacy_ner"
    payload = {
        "query": query,
        "components": components,
        "lang": "en"
    }
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # Make the POST request to the NER/EL pipeline
        response = requests.post(
            os.environ.get('LFQA_URL', default='http://porque.cs.upb.de/porque-neamt/custom-pipeline'),
            headers=headers,
            json=payload,
            timeout=600
        )
        response.raise_for_status()  # Raise an error for bad HTTP responses (4xx, 5xx)

    except requests.exceptions.RequestException as e:
        # Handle network errors or bad responses
        logging.error(f"NER/EL pipeline request failed: {e}")
        return []
    except Exception as e:
        logging.error(f"Error from: {os.environ.get('LFQA_URL', default='http://porque.cs.upb.de/porque-neamt/custom-pipeline')} {e}")


    try:
        link_info = response.json()
        # entities = []

        # Handle the case where response is a list
        if isinstance(link_info, list):
            link_info = link_info[0]

        # If no 'ent_mentions' key, return an empty list
        if 'ent_mentions' not in link_info:
            logging.info("No entity mentions found in the response.")
            return entities

        # Process the entities
        for ent in link_info['ent_mentions']:
            surface_form = ent.get('surface_form', "")

            # Ensure valid entity and check if it appears in the query using a helper function
            if surface_form and surface_form not in {'"', '”'} and check_if_synonyms_in_sentence(query, surface_form):
                entities.add(surface_form)
                logging.info(f"Entity found from NER tool: {surface_form}")
            else:
                logging.info(f"Entity mismatch from NER tool: {surface_form}")

        return entities

    except ValueError as e:
        # Handle JSON decoding errors (malformed JSON in response)
        logging.error(f"Failed to parse JSON response: {e}")
        dump_malformed_json(response.text)  # Dump raw response to file for debugging
        return []


def query_wikidata(search_term):
    """
    Queries the Wikidata SPARQL endpoint for a relation based on the provided search term.

    Args:
        search_term (str): The search term to look up relations for.

    Returns:
        list: A list of relations (URIs) that match the search term in the SPARQL query.
    """
    # Define the SPARQL query with the search term
    sparql_query = f"""
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?relation 
    WHERE {{
      ?relation a wikibase:Property ;
                 rdfs:label ?label .
      FILTER (LANG(?label) = "en" && str(?label) = "{search_term}")
    }}
    """

    # Wikidata SPARQL endpoint (update this based on your environment)
    wikidata_endpoint = "https://wikidata.data.dice-research.org/sparql"

    # Set up the request headers
    headers = {
        "Accept": "application/sparql-results+json"
    }

    try:
        # Make the HTTP request to the Wikidata SPARQL endpoint
        response = requests.post(wikidata_endpoint, headers=headers, data={"query": sparql_query}, timeout=10)

        # Check if the request was successful (status code 200)
        response.raise_for_status()

        # Parse the JSON response
        data = response.json()

        # Ensure 'results' and 'bindings' are present in the response
        if "results" in data and "bindings" in data["results"]:
            results = data["results"]["bindings"]

            # Extract the relations from the response
            relations = [result["relation"]["value"] for result in results if "relation" in result]

            # Log the number of relations found
            logging.info(f"Found {len(relations)} relations for search term '{search_term}'.")
            return relations

        else:
            logging.warning("No results or bindings found in the response.")
            return []

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
        return []
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout occurred: {timeout_err}")
        return []
    except requests.exceptions.RequestException as req_err:
        logging.error(f"An error occurred during the request: {req_err}")
        return []
    except ValueError as json_err:
        logging.error(f"Failed to parse JSON response: {json_err}")
        return []


def get_relations_between_entities(entities):
    """
    Fetch relations between all possible combinations of entities (pairs).

    Args:
        entities (list): List of entity names (e.g., ["entity1", "entity2", "entity3"]).

    Returns:
        dict: A dictionary with entity pairs as keys and their relations as values.
    """
    relations = []

    # Generate all possible pairs of entities using itertools.combinations
    entity_pairs = list(itertools.combinations(entities, 2))

    # Iterate over each entity pair
    for entity1, entity2 in entity_pairs:
        # Query Wikidata or any other service for relations between entity1 and entity2
        # Here, we assume there's a function that queries relations for a pair of entities.
        relation = query_wikidata_for_relation(entity1, entity2)
        # Format the result as a dictionary and append to the relations list
        relations.append({
            'head': entity1,
            'tail': entity2,
            'type': relation  # Assuming relation returns a string or list of relations
        })
        # Store the relations in a dictionary using (entity1, entity2) as the key
        # relations[(entity1, entity2)] = relation

    return relations

def query_wikidata_for_relation(entity1, entity2):
    # SPARQL endpoint URL for Wikidata
    url = "https://wikidata.data.dice-research.org/sparql"

    # SPARQL query to find relations between the two entities
    query = f"""
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?property ?propertyLabel
    WHERE {{
      {{ 
        wd:{entity1} ?property wd:{entity2} .
      }}
      UNION
      {{ 
        wd:{entity2} ?property wd:{entity1} .
      }}
    }}
    """

    # HTTP headers for the POST request
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Python script"
    }

    # Payload for the POST request (the query wrapped in the 'query' field)
    payload = {
        'query': query,
        'format': 'json'
    }

    # Send the POST request
    response = requests.get(url, params="query=" + query)

    # response = requests.get(url, data=payload)

    # Check if the response status is OK
    if response.status_code == 200:
        # Convert the response to JSON
        results = json.loads(response.content)
        logging.info(results)

        # Extract and return the relations - INFO:root:{'head': {'vars': ['property']}, 'results': {'bindings': [{'property': {'type': 'uri', 'value': 'http://www.wikidata.org/prop/direct/P17'}}]}}
        relations = []
        relationLabels = []
        for result in results["results"]["bindings"]:
            if "property" in result.keys():
                relation = result["property"]["value"]
            else:
                # relation = result["relationLabel"]["value"]
                logging.error("relation IRI missing")

            if "propertyLabel" in result.keys():
                label = result["propertyLabel"]["value"]
            else:
                logging.error("relation label missing")
                label = str(relation)

            relations.append(relation)
            relationLabels.append(label)
        return relations, relationLabels
    else:
        # Handle errors (e.g., bad response)
        return f"Error: Unable to fetch data, status code {response.status_code}"


# Load spaCy model for better semantic similarity
nlp = spacy.load("en_core_web_md")  # Change to "de_core_news_md" for German

def check_best_similarity(label, list_of_heads, threshold=80):
    """
    Checks for the best similarity match between a given label and a list of head entities.

    Args:
        label (str): The label to match against.
        list_of_heads (list): A list of heads, where each head is a tuple and the third element (index 2) is the value to return if matched.
        threshold (int): Minimum similarity score to consider a match.

    Returns:
        str: The best matching head's third element (index 2). Returns None if no match is found.
    """
    label = str(label).lower()

    best_match = None
    best_score = 0

    for head in list_of_heads:
        head_label = str(head[0]).lower()
        head_value = head[2]

        if not head_value.strip():
            continue

        # 1. Exact Match
        if head_label == label:
            return head_value

        # 2. Fuzzy Matching (Levenshtein Similarity)
        fuzzy_score = fuzz.ratio(head_label, label)
        if fuzzy_score > best_score and fuzzy_score >= threshold:
            best_match, best_score = head_value, fuzzy_score

        # 3. Substring Match (Basic Heuristic)
        if head_label in label or label in head_label:
            return head_value

        # 4. Semantic Similarity using spaCy
        doc1, doc2 = nlp(head_label), nlp(label)
        semantic_score = doc1.similarity(doc2)
        if semantic_score > best_score and semantic_score >= 0.7:  # Adjust threshold as needed
            best_match, best_score = head_value, semantic_score * 100  # Normalize score

    # Return best match found or None if no match
    if best_match:
        return best_match

    # Log a warning if no match is found
    logging.warning(f"No good match found for '{label}' in the provided list.")
    return None


# def check_best_similarity(label, list_of_heads):
#     """
#     Checks for the best similarity match between a given label and a list of head entities.
#
#     Args:
#         label (str): The label to match against.
#         list_of_heads (list): A list of heads, where each head is a tuple and the third element (index 2) is the value to return if matched.
#
#     Returns:
#         str: The best matching head's third element (index 2). Returns the first element's third value by default if no match is found.
#     """
#     # Ensure label is in lowercase for consistent comparison
#     label = str(label).lower()
#
#     # Iterate through the list of heads
#     for head in list_of_heads:
#         # Ensure head[2] is not an empty string
#         if not str(head[2]).strip():
#             continue
#
#         # Check for direct match (case-insensitive)
#         if str(head[0]).lower() == label:
#             return head[2]
#
#         # Check for a contextual match (if available)
#         if is_contextual_match(head[0], label):
#             return head[2]
#
#     # If no match found, return the third element of the first head as a default
#     if list_of_heads:
#         return list_of_heads[0][2]
#
#     # Optional: In case of an empty list, handle gracefully
#     logging.warning("Empty list_of_heads provided.")
#     return None  # Or some other default value, depending on your needs


def is_valid_json(line):
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        print("invalid:" + line)
        return False

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj

def save_entities_relations(path, data_dict, args):
    # Open the file for writing
    existing_keys = load_existing_keys(path)
    # Filter out new items
    # Using set difference to identify new keys first
    new_keys = set(data_dict.keys()).difference(existing_keys)
    new_items = {key: data_dict[key] for key in new_keys}

    if new_items:
        with file_write_lock:  # Lock ensures exclusive access during file writing
            with open(path, 'a') as file:
                # Serialize the dictionary to JSON and write it to the file add_backslash_to_quotes
                for item in new_items:
                    if not isinstance(new_items[item], list):
                        if item == '':
                            continue
                        dict1 = {
                            item: new_items[item]
                        }
                    else:
                        dict1 = {
                            item: new_items[item][0]
                        }
                    json.dump(dict1, file)
                    file.write('\n')  # Add newline character after each JSON object

def save_processed_triples_data(path, triples):
    # Open the file for writing
    with open(path, 'a') as file:
        # Serialize the list to JSON and write it to the file
        for triple in triples:
            dict1 = {
                "article_id": triple[0],
                "triple": {
                    triple[1]
                },
                "subject": triple[2],
                "predicate": triple[3],
                "object": triple[4],
                "claim": triple[5],
                "original_claim": triple[6],
                "id": triple[7]
            }
            json.dump(dict1, file, default=serialize_sets)
            file.write('\n')  # Add newline character after each JSON object
            logging.info("saving:" + triple[4])
    logging.info("saving triples")


def get_ent_IRIs_from_api(query, head, tail, entities_dict, error_ent_dict):
    # components = "mgenre_el"
    components = os.environ.get('EL_TOOL', default='mgenre_el') #"babelscape_ner"
    payload = {
        "query": query,
        "components": components,
        "lang": "en",
        "mg_num_return_sequences": 5,
        "ent_mentions": []
    }
    headers = {
        'Content-Type': 'application/json'
    }

    # Add entity mentions to the payload
    for entity in [(head, "head"), (tail, "tail")]:
        entity_str, entity_name = entity
        try:
            start_index = query.index(entity_str)
            end_index = start_index + len(entity_str)
            payload["ent_mentions"].append({
                "start": start_index,
                "end": end_index,
                "surface_form": entity_str
            })
        except ValueError:
            logging.error(f"{entity_name.capitalize()} entity '{entity_str}' not found in query.")
            return '', '', False  # Early return if head or tail not in query

    # Return cached IRIs if available
    if (head in entities_dict or head in error_ent_dict) and (tail in entities_dict or tail in error_ent_dict):
        logging.info("Returning cached IRIs from Dict.")
        head_IRI = ""
        tail_IRI = ""
        if head in entities_dict:
            head_IRI = entities_dict[head]
        else:
            head_IRI = error_ent_dict[head]

        if tail in entities_dict:
            tail_IRI = entities_dict[tail]
        else:
            tail_IRI = error_ent_dict[tail]

        return head_IRI, tail_IRI, False

    try:
        logging.error(f"Querying API to get IRIs: {os.environ.get('LFQA_URL', default='http://porque.cs.upb.de/porque-neamt/custom-pipeline')}")
        response = requests.post(os.environ.get('LFQA_URL', default='http://porque.cs.upb.de/porque-neamt/custom-pipeline'), headers=headers,
                                 json=payload, timeout=600)
        response.raise_for_status()  # Raises an error for HTTP status codes 4xx or 5xx
        link_info = response.json()

        if isinstance(link_info, list) and len(link_info) > 0:
            link_info = link_info[0]

        # Check if response contains expected structure
        if "ent_mentions" not in link_info or len(link_info["ent_mentions"]) < 2:
            if len(link_info) == 0:
                logging.info("0 entities found")
            logging.info("Invalid response structure received from API.")
            return '', '', False

        # Extract best matching IRI for head and tail
        head_IRI = check_best_similarity(label=head, list_of_heads=link_info['ent_mentions'][0]['link_candidates'])
        tail_IRI = check_best_similarity(label=tail, list_of_heads=link_info['ent_mentions'][1]['link_candidates'])

        # Cache results
        if head_IRI:
            entities_dict[head] = head_IRI
        elif response.status_code == 200:
            error_ent_dict[head] = head_IRI
        if tail_IRI:
            entities_dict[tail] = tail_IRI
        elif response.status_code == 200:
            error_ent_dict[tail] = tail_IRI

        logging.error("IRIs returned from API: head-"+str(head_IRI)+"-tail-"+str(tail_IRI))
        return head_IRI, tail_IRI, True

    except requests.exceptions.RequestException as e:
        logging.error("Service unresponsive: http://porque.cs.upb.de/porque-neamt/custom-pipeline")
        logging.error(f"Request error: {e}")
        dump_malformed_json(str(response.text))
    except (KeyError, IndexError, TypeError) as e:
        logging.error("Error processing API response.")
        logging.error(f"Error details: {e}")
        logging.error(f"Content: {link_info['ent_mentions']}")
        dump_malformed_json(str(response.text))
    except:
        logging.error("entity linking failed:"+str(response.text))
        dump_malformed_json(str(response))
    # Return empty IRIs if anything fails
    return '', '', False
# bulk_triples should be false for API
def start_entity_linking(lines, args, entities_dict, error_ent_dict, relations_dict, bulk_triples=True):
    dataset_type = args["--dataset_type"]

    file_name = f"{dataset_type}_{args['--output_IRIs_file']}"
    dataset_path = args['--dataset_path']
    triples = []
    entities_cache_updated = False
    relations_cache_updated = False
    logging.info("\n\nEL starts here! output file path: "+str(file_name)+". \nDataset path:"+str(dataset_path))
    for line in lines:
        query = line["claim"]
        original_claim = line["original_claim"]
        article_id = line.get("article_id", line["id"])

        # Skip if already processed
        claim_key = f"{article_id}-{original_claim}"
        if claim_key in processed_claims and bulk_triples == True:
            logging.info("Already processed: " + claim_key)
            continue

        # logging.info("Not already processed: " + claim_key)

        # Check for required fields
        if not line.get("head") or not line.get("type") or not line.get("tail"):
            logging.info("Head/type/tail empty: " + claim_key)
            continue

        head, tail, rel = line["head"], line["tail"], line["type"]

        # Ensure head and tail are in the query
        if head not in query or tail not in query:
            logging.info("Head/tail not in query: " + claim_key)
            continue

        head_IRI, tail_IRI, entities_cache_updated = get_ent_IRIs_from_api(query, head, tail, entities_dict, error_ent_dict)

        if len(rel) > 200:
            print("Invalid relation length. Exiting.")
            exit(1)

        if head_IRI and tail_IRI:
            # Check if relation is already known
            relation = relations_dict.get(rel)
            if relation is None:
                relations_cache_updated = True
                logging.info("Querying Wikidata for relation: " + rel)
                relation = query_wikidata(rel)

                if relation:
                    relations_dict[rel] = relation[0] if (isinstance(relation,list) and len(relation)>0) else relation
                else:
                    #needs testing here
                    relations, labls =  None, None # query_wikidata_for_relation(head_IRI,tail_IRI)
                    if relations:
                        for rel1, lab1 in zip(relations,labls):
                            if rel1 and lab1:
                                relations_dict[lab1] = rel1
                            else:
                                logging.error(
                                    "problem in relations " + str(rel1) + "-" + str(lab1))
                    else:
                        # Fallback to a new relation, if no relation found
                        relation = f"http://dice-research.org/P-{rel.replace(' ', '_')}"
                        relations_dict[rel] = relation
            else:
                logging.info("Using cached relation: " + str(relation))
            if isinstance(relation, list) and len(relation)>0:
                relation = relation[0]
            # Prepare the triple if the relation is valid
            if relation and relation not in {'N/A', 'Bad Request'}:
                relation = relation.strip("[]'\"")  # Clean up the relation
                triple = (
                f"https://www.wikidata.org/wiki/{head_IRI}", relation, f"https://www.wikidata.org/wiki/{tail_IRI}")
                triples.append([article_id, triple, head, rel, tail, query, original_claim, line["id"]])

                logging.info("Processed triple: " + str(triple))
                logging.info('Length of triples in cache: ' + str(len(triples)))

                # Save triples periodically  ...after that empty
                if triples and bulk_triples:
                    output_file_path = f"{dataset_path}linking/{file_name}"
                    save_processed_triples_data(output_file_path, triples)
                    processed_claims.append(claim_key)
                    triples.clear()

                    # Save entities and relations
                # if entities_cache_updated:
                #     logging.info("Saving updated entites in cache")
                #     save_entities_relations(f"{dataset_path}linking/{args['--entities_dict']}", entities_dict, args)
                #     save_entities_relations(f"{dataset_path}linking/{args['--error_ent_dict']}", error_ent_dict, args)
                #     logging.info("Saving completed")
                # if relations_cache_updated:
                #     logging.info("Saving updated relations in cache")
                #     save_entities_relations(f"{dataset_path}linking/{args['--relations_dict']}", relations_dict, args)
                #     logging.info("Saving completed")
                #
                #     logging.info("Triples processed: " + str(len(triples)))

                    # triples.clear()  # Clear the list after saving
                # else:
                #     if not bulk_triples and triples:
                #         return triples, entities_dict, relations_dict

                    logging.info("No triples to process.")
            else:
                logging.info("Saved invalid triples.")
        else:
            logging.info("Head or tail values not found for this claim.")
    if entities_cache_updated:
        logging.info("Saving updated entites in cache")
        save_entities_relations(f"{dataset_path}linking/{args['--entities_dict']}", entities_dict, args)
        save_entities_relations(f"{dataset_path}linking/{args['--error_ent_dict']}", error_ent_dict, args)
        logging.info("Saving completed")
    if relations_cache_updated:
        logging.info("Saving updated relations in cache")
        save_entities_relations(f"{dataset_path}linking/{args['--relations_dict']}", relations_dict, args)
        logging.info("Saving completed")

        logging.info("Triples processed: " + str(len(triples)))
    return triples, entities_dict, relations_dict, error_ent_dict


def create_new_file_if_not_exist(file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as file:
            file.write('')
        print(f"New file '{file_name}' created.")
    else:
        logging.info(f"File '{file_name}' already exists.")
def add_claim_in_processed_claims(id,sent):
    processed_claims.append(id+"-"+sent)
def read_processed_claims(output_file_path, tag='original_claim'):

    create_new_file_if_not_exist(output_file_path)
    claims = []
    id_claims = []
    # Open the file for reading
    with jsonlines.open(output_file_path, 'r') as file:
        # Load the JSON data from the file
        for claim in file:
            claims.append(claim.get(tag))
            val = claim.get('article_id', claim.get('id'))
            value = str(val) if isinstance(val, int) else val
            id_claims.append(value+"-"+claim.get(tag))

    # Now 'data' contains a Python dictionary representing the JSON content
    print(str(len(claims)))
    global  processed_claims
    processed_claims = id_claims
    return processed_claims
# get already processed relations and entities

def load_existing_keys(filepath):
    """Load existing keys from a JSONL file to check for duplicates."""
    keys = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                keys.update(entry.keys())
    return keys


def get_processed_data(dataset_path, entities_file,error_ent_dict_file, relations_file):
    logging.info("getting processed claims")
    path = dataset_path


    # Define the file path
    entities_file_path = path + 'linking/' + entities_file
    create_new_file_if_not_exist(entities_file_path)
    with open(entities_file_path, 'r') as file:
        entity_lines = file.readlines()

    processed_entities = []
    for ll in entity_lines:
        # print(ll)
        # logging.info(str(ll))
        dic_ent = json.loads(ll)
        # j_res = repr(dic_ent.keys())
        processed_entities.append(dic_ent)
        entities_dict[list(dic_ent.keys())[0]] = dic_ent[list(dic_ent.keys())[0]]

    # Define the file path
    error_ent_dict_file_path = path + 'linking/' + error_ent_dict_file
    create_new_file_if_not_exist(error_ent_dict_file_path)
    with open(error_ent_dict_file_path, 'r') as file:
        entity_lines = file.readlines()

    processed_entities = []
    for ll in entity_lines:
        # print(ll)
        # logging.info(str(ll))
        dic_ent = json.loads(ll)
        # j_res = repr(dic_ent.keys())
        processed_entities.append(dic_ent)
        error_ent_dict[list(dic_ent.keys())[0]] = dic_ent[list(dic_ent.keys())[0]]

    # Define the file path
    relations_file_path = path + 'linking/' + relations_file
    create_new_file_if_not_exist(relations_file_path)
    with open(relations_file_path, 'r') as file:
        relation_lines = file.readlines()

    processed_relations = []
    for ll in relation_lines:
        # print("relation:" + ll)
        dic_ent = json.loads(ll)
        # print("actual key:" + str(dic_ent.keys()))
        processed_relations.append(dic_ent)
        relations_dict[list(dic_ent.keys())[0]] = dic_ent[list(dic_ent.keys())[0]]


    return entities_dict, relations_dict, error_ent_dict



    # for triple in result:
    #     print(triple)
# parser.add_argument("--path",  default='/upb/users/u/uqudus/profiles/unix/cs/NEBULA/TripleExtraction/rebel_output/FEVER/output/output_train/')# parser.add_argument("--path",  default='/upb/users/u/uqudus/profiles/unix/cs/NEBULA/TripleExtraction/rebel_output/FEVER/output/output_train/')
