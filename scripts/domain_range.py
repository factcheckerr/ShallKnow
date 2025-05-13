import json
import ast
import urllib3
from collections import defaultdict
import re
import os
import requests
from rdflib import Graph, URIRef, Literal
from SPARQLWrapper import SPARQLWrapper, JSON
import pickle

global entities_class_hirarchy
entities_class_hirarchy = {}

def save_class_hierarchy():
    # Save cache to disk
    with open('class_hierarchy_cache.pkl', 'wb') as cache_file:
        pickle.dump(entities_class_hirarchy, cache_file)

def load_class_hierarchy():
    global entities_class_hirarchy
    # Load cache from disk
    with open('class_hierarchy_cache.pkl', 'rb') as cache_file:
        entities_class_hirarchy = pickle.load(cache_file)


global predicate_domain_mapping
predicate_domain_mapping = defaultdict(dict)

global predicate_range_mapping
predicate_range_mapping = defaultdict(dict)

domain_cache = {}
range_cache = {}


def save_dictionary(file_path):
    """
    Saves a dictionary to a file in JSON format.

    Args:
        file_path (str): Path to the file where the dictionary will be saved.
        dictionary (dict): Dictionary to save. Values must be lists.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(entities_class_hirarchy, file, indent=4)
        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving dictionary: {e}")


def load_dictionary(file_path):
    """
    Loads a dictionary from a file in JSON format.

    Args:
        file_path (str): Path to the file from which the dictionary will be loaded.

    Returns:
        dict: The loaded dictionary.
    """
    ensure_file_exists(file_path)
    if os.path.getsize(file_path) == 0:
        print(f"File is empty: {file_path}")
        return {}
    try:
        with open(file_path, 'r') as file:
            dictionary = json.load(file)
        print(f"Dictionary successfully loaded from {file_path}")
        return dictionary
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        return {}


def parse_jsonl_file(filepath,  error_log_filepath="error_parsing_dr.txt"):
    triples_data = []

    def clean_and_parse_item(item):
        """Clean and parse an invalid stringified dictionary."""
        try:
            # Replace single quotes with double quotes for JSON compatibility
            item = re.sub(r"'", '"', item)

            # Replace Python-style set notation with JSON-compatible list notation
            item = re.sub(r"{\((.*?)\)}", r'[[\1]]', item)

            # Replace Python-style tuples with JSON-compatible lists
            item = re.sub(r"\((.*?)\)", r'[\1]', item)

            # Load the cleaned string as JSON
            return json.loads(item)
        except json.JSONDecodeError as e:
            with open(error_log_filepath, "a") as error_file:
                error_file.write(f"Failed to clean/parse item: {item}, Error: {e}\n")
            return None


    with open(filepath, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            triples_field = data.get('triples', [])

            # Normalize triples_field to a list of dictionaries
            if isinstance(triples_field, dict):
                triples_list = [triples_field]
            elif isinstance(triples_field, list):
                triples_list = []
                for item in triples_field:
                    if isinstance(item, str):
                        parsed_item = clean_and_parse_item(item)
                        if parsed_item:
                            triples_list.append(parsed_item)
                    elif isinstance(item, dict):
                        triples_list.append(item)
                    else:
                        with open(error_log_filepath, "a") as error_file:
                            error_file.write(f"Unexpected format in triples list: {item}\n")
            else:
                with open(error_log_filepath, "a") as error_file:
                    error_file.write(f"Unexpected triples field format: {triples_field}\n")
                continue

            # Process each triple in the normalized triples list
            for triple_data in triples_list:
                try:
                    triple_set = triple_data.get('triple', None)

                    # Normalize triple_set to a list
                    if isinstance(triple_set, set):
                        triple_set = list(triple_set)
                    elif isinstance(triple_set, list):
                        pass  # Already a list
                    else:
                        print(f"Unexpected triple_set format: {data}")
                        with open(error_log_filepath, "a") as error_file:
                            error_file.write(f"Unexpected format in triples list: {data}\n")
                        continue

                    for triple in triple_set:
                        # Validate triple is a tuple or list of size 3
                        if isinstance(triple, (tuple, list)) and len(triple) == 3:
                            subject_uri, predicate_uri, object_uri = triple
                            subject = triple_data.get('subject', '')
                            predicate = triple_data.get('predicate', '')
                            obj = triple_data.get('object', '')

                            # Append the parsed data
                            triples_data.append({
                                'subject': subject,
                                'predicate': predicate,
                                'object': obj,
                                'subject_uri': subject_uri,
                                'predicate_uri': predicate_uri,
                                'object_uri': object_uri
                            })
                        else:
                            with open(error_log_filepath, "a") as error_file:
                                error_file.write(f"Invalid triple format: {triple}\n")
                except (ValueError, TypeError) as e:
                        with open(error_log_filepath, "a") as error_file:
                            error_file.write(f"Failed to parse triple_data: {triple_data}, Error: {e}\n")
    return triples_data

def load_triples_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            triples = json.load(file)
        print(f"Loaded {len(triples)} triples from {file_path}.")
        return triples
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []


def determine_known_and_unknown_predicates(triples_data):
    known_predicates = set()
    unknown_predicates = set()

    for triple in triples_data:
        if isinstance(triple, dict):
            predicate_uri = triple['predicate_uri']
        else:
            predicate_uri = triple[1]
        if 'wikidata.org' in predicate_uri:
            known_predicates.add(predicate_uri)
        else:
            unknown_predicates.add(predicate_uri)
    return known_predicates, unknown_predicates


def get_domain(predicate):
    if predicate in domain_cache:
        return domain_cache[predicate]
    predicate_id = predicate.split('/')[-1].replace(">","")
    wd_predicate = f"wd:{predicate_id}"
    query = ("PREFIX p: <http://www.wikidata.org/prop/> "
             "PREFIX ps: <http://www.wikidata.org/prop/statement/> "
             "PREFIX pq: <http://www.wikidata.org/prop/qualifier/> "
             "PREFIX wd: <http://www.wikidata.org/entity/> "
             "SELECT DISTINCT ?domain WHERE {" + str(wd_predicate) + " p:P2302 ?o . "
                                                                       "?o ps:P2302 wd:Q21503250 . "
                                                                       "?o pq:P2308 ?domain .}")
    # print(f"queryyyyyyy: {query}")
    results = run_sparql_query(query)
    # print(results, "*****")
    if not results:
        print(f"No domain found for predicate: {predicate}")
        domain_cache[predicate] = []
        return []

    domain_cache[predicate] = [result['domain']['value'] for result in results['results']['bindings']]
    return domain_cache[predicate]


def get_range(predicate):
    if predicate in range_cache:
        return range_cache[predicate]
    predicate_id = predicate.split('/')[-1].replace(">","")
    wd_predicate = f"wd:{predicate_id}"
    query = "PREFIX p: <http://www.wikidata.org/prop/> PREFIX ps: <http://www.wikidata.org/prop/statement/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX wd: <http://www.wikidata.org/entity/> SELECT DISTINCT ?range WHERE {"+str(
        wd_predicate)+" p:P2302 ?o . ?o ps:P2302 wd:Q21510865 . ?o pq:P2308 ?range .}"
    # print(f"queryyyyyyy: {query}")
    results = run_sparql_query(query)
    # print(results, "*****")
    if not results:
        print(f"No range found for predicate: {predicate}")
        range_cache[predicate] = []
        return []

    range_cache[predicate] = [result['range']['value'] for result in results['results']['bindings']]
    return range_cache[predicate]


def run_sparql_query(query_string, endpoint_url="https://query.wikidata.org/sparql"):
    print("Executing SPARQL query:", query_string)
    http = urllib3.PoolManager()
    request_header = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    try:
        response = http.request('GET', endpoint_url, fields={'query': query_string}, headers=request_header)

        if response.status != 200:
            print(f"SPARQL query failed with status code: {response.status}")
            return {}

        result = json.loads(response.data.decode('utf-8'))
        return result
    except Exception as e:
        print(f"An error occurred while executing the SPARQL query: {e}")
        exit(1)
        return {}


def create_class_hierarchy_sparql_query(uris):
    """
    Create a SPARQL query to fetch the class hierarchy for a list of URIs.
    """
    entities_filter = " ".join([f"{uri}" for uri in uris])

    sparql_query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?entity ?class WHERE {{
        VALUES ?entity {{ {entities_filter} }}
        ?entity wdt:P31*/wdt:P279* ?class.
    }}
    """
    return sparql_query

def batch_query_class_hierarchy(uris, batch_size=100):
    """
    Query the class hierarchy of entities in batches to avoid sending too many requests to Wikidata.
    """
    all_results = []

    # Split URIs into batches
    for i in range(0, len(uris), batch_size):
        batch = list(uris)[i:i + batch_size]

        sparql_query = create_class_hierarchy_sparql_query(batch)

        try:
            data = run_sparql_query(sparql_query)

            # Process the results and store the class hierarchy for each URI
            result_dict = {}
            for item in data['results']['bindings']:
                entity = item['entity']['value']
                class_uri = item.get('class', {}).get('value', None)
                if class_uri:
                    if entity in result_dict:
                        result_dict[entity].append(class_uri)
                    else:
                        result_dict[entity] = [class_uri]
                    # result_dict[entity] = class_uri

            all_results.append(result_dict)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching class hierarchy for batch: {e}")
            exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)

    # Combine all results and return
    combined_results = {}
    for result in all_results:
        combined_results.update(result)

    return combined_results


def query_class_hierarchy(uri):
    entity_id = str(uri.split('/')[-1]).replace(">","").replace("<","")
    wd_entity = f"wd:{entity_id}"

    # query = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> SELECT ?class ?superclass WHERE {" + str(
    #     wd_entity) + "  wdt:P31 ?class . ?class wdt:P279* ?superclass . }"
    query = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
             "PREFIX wd: <http://www.wikidata.org/entity/> "
             "SELECT ?class  "
             "WHERE {" + str(wd_entity) + " wdt:P31*/wdt:P279* ?class. }")

    # print(f"queryyyyyyy: {query}")
    results = run_sparql_query(query)
    # print(results, "*****")
    if not results:
        print(f"No class found for uri: {uri}")
        return []

    return [result['class']['value'] for result in results['results']['bindings']]


def load_processed_triples(file_path='processed_triples_test.nt'):
    processed_triples = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.split(" ")
                # if not line or line.startswith('#'):
                #     continue
                # triple_part = line[:-1].strip()
                # s, p, o = triple_part.split()
                processed_triples.add((line[0], line[1], line[2]))

    except FileNotFoundError:
        print(f"'{file_path}' not found. Will create a new file when saving.")

    return processed_triples



def save_processed_triples(processed_triples, processed_file_path='processed_triples.nt'):
    with open(processed_file_path, 'w') as file:
        for triple in processed_triples:
            s, p, o = triple
            file.write(f"{s} {p} {o} .\n")


def parse_nt_file(file_path):
    """Parse the NT file to create a mapping of subclassOf relationships."""
    class_mappings = {}
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) == 4 and parts[1] == 'rdfs:subclassOf':
                    subclass = parts[0].strip('<>')
                    superclass = parts[2].strip('<>')
                    if superclass not in class_mappings:
                        class_mappings[superclass] = []
                    class_mappings[superclass].append(subclass)
    return class_mappings

def create_mapping_from_json_and_nt(json_file_path, nt_file_path):
    """Create a mapping using the JSON and NT file."""
    # Ensure both files exist
    global  predicate_domain_mapping, predicate_range_mapping
    if not (os.path.exists(json_file_path) and os.path.exists(nt_file_path)):
        print("One or both files do not exist.")
        predicate_domain_mapping = defaultdict(dict)
        predicate_range_mapping = defaultdict(dict)
        ensure_file_exists(json_file_path)
        ensure_file_exists(nt_file_path)
        return
        # return {}, {}

    # Parse the NT file
    class_mappings = parse_nt_file(nt_file_path)

    # Parse the JSON file
    ensure_file_exists(json_file_path)
    with open(json_file_path, 'r') as file:
        try:
            data = json.load(file)
            if isinstance(data, list):
                # predicate_domain_mapping = {}
                # predicate_range_mapping = {}
                for item in data:
                    predicate = item.get("predicate")
                    identifier = item.get("domain")  # Assuming 'domain' maps to 'identifier'

                    if predicate and identifier:
                        # Get classes for the identifier from the NT file mapping
                        classes = class_mappings.get(identifier, [])
                        predicate_domain_mapping[predicate] = {
                            "identifier": identifier,
                            "classes": classes
                        }
                for item in data:
                    predicate = item.get("predicate")
                    identifier = item.get("range")  # Assuming 'domain' maps to 'identifier'

                    if predicate and identifier:
                        # Get classes for the identifier from the NT file mapping
                        classes = class_mappings.get(identifier, [])
                        predicate_range_mapping[predicate] = {
                            "identifier": identifier,
                            "classes": classes
                        }

                # return predicate_domain_mapping, predicate_range_mapping
            else:
                print("Invalid JSON format: Expected a list of objects.")

        except json.JSONDecodeError as e:
            print(f"Error reading JSON: {e}")
            exit(1)
    return



def assign_single_identifier(predicate, classes, is_domain=True):
    global dp_counter, rp_counter
    mapping_dict = predicate_domain_mapping if is_domain else predicate_range_mapping

    predicate_id = predicate.split('/')[-1].replace(">","")  # Extract the predicate ID (e.g., P360)
    identifier = f"D{predicate_id}" if is_domain else f"R{predicate_id}"

    if predicate not in mapping_dict:
        mapping_dict[predicate] = {"identifier": identifier, "classes": classes}
    else:
        existing_classes = mapping_dict[predicate]["classes"]
        mapping_dict[predicate]["classes"] = list(set(existing_classes).union(set(classes)))
    if mapping_dict[predicate]["identifier"]!= identifier:
        mapping_dict[predicate]["identifier"] = identifier
    return mapping_dict[predicate]["identifier"]


def class_mapping(predicate_class_mappings_file_path='test_predicate_class_mappings.nt'):
    print("class mapping and saving to n triples formattttttttt")
    predicate_class_mappings = {}

    for predicate_uri, domain_info in predicate_domain_mapping.items():
        if predicate_uri not in predicate_class_mappings:
            predicate_class_mappings[predicate_uri] = {"domain_classes": {}, "range_classes": {}}
        for cls in domain_info["classes"]:
            predicate_class_mappings[predicate_uri]["domain_classes"][cls] = domain_info["identifier"]

    for predicate_uri, range_info in predicate_range_mapping.items():
        if predicate_uri not in predicate_class_mappings:
            predicate_class_mappings[predicate_uri] = {"domain_classes": {}, "range_classes": {}}
        for cls in range_info["classes"]:
            predicate_class_mappings[predicate_uri]["range_classes"][cls] = range_info["identifier"]

    lines = set()
    ensure_file_exists(predicate_class_mappings_file_path)
    with open(predicate_class_mappings_file_path, "r") as file:
        for line in file:
            lines.add(line)

    count = 0
    for predicate_uri, class_info in predicate_class_mappings.items():
        for cls, dp in class_info["domain_classes"].items():
            if f"<{cls}> rdfs:subclassOf <http://dice-research.org/{dp}> .\n" not in lines:
                count += 1
                if str(dp).__contains__("dice-research"):
                    lines.add(f"<{cls}> rdfs:subclassOf <{dp}> .\n")
                else:
                    lines.add(f"<{cls}> rdfs:subclassOf <http://dice-research.org/{dp}> .\n")

        for cls, rp in class_info["range_classes"].items():
            if f"<{cls}> rdfs:subclassOf <http://dice-research.org/{rp}> .\n" not in lines:
                count += 1
                if str(rp).__contains__("dice-research"):
                    lines.add(f"<{cls}> rdfs:subclassOf <{rp}> .\n")
                else:
                    lines.add(f"<{cls}> rdfs:subclassOf <http://dice-research.org/{rp}> .\n")
    if count > 0:
        with open(predicate_class_mappings_file_path, "w") as mapping_file:
            mapping_file.write("".join(lines))
    print(f"Class mappings successfully written to {predicate_class_mappings_file_path}.")



def ensure_file_exists(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file
        with open(file_path, 'w') as file:
            pass  # Create an empty file
        print(f"File '{file_path}' created.")
    else:
        print(f"File '{file_path}' already exists.")
def save_sparql_queries(output_file, triples_data, pred = None):
    # queries = []
    ensure_file_exists(output_file)
    subject_predicate_set = set()
    object_predicate_set = set()
    queries_set = set()

    # Read existing queries from file
    with open(output_file, "r") as file:
        for line in file:
            queries_set.add(line.strip())
            # queries.append(line)
            parts = line.split(" ")
            if pred is not None:
                if str(parts[2]).__contains__("DP"):
                    subject_predicate_set.add((parts[0],pred))
                elif str(parts[2]).__contains__("RP"):
                    object_predicate_set.add((parts[0],pred))

    len_queries = len(triples_data)
    count = 0
    new_queries = []

    # Process triples
    # with open(output_file, "a") as file:
    for triple in triples_data:
        if isinstance(triple, dict):
            predicate_uri = triple['predicate_uri']
            subject_uri = f"<{triple['subject_uri']}>"
            object_uri = f"<{triple['object_uri']}>"
        else:
            subject_uri = triple[0]
            predicate_uri = triple[1]
            object_uri = triple[2]
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} of {len_queries} triples. Queries saved")
        if (subject_uri, predicate_uri) not in subject_predicate_set:
            subject_predicate_set.add((subject_uri, predicate_uri))
            # Handle domain insertion
            if predicate_uri in predicate_domain_mapping:
                domain_identifier = predicate_domain_mapping[predicate_uri].get("identifier", None)
                if domain_identifier:
                    sparql_query = f"{subject_uri} rdf:type <http://dice-research.org/{domain_identifier}> .\n"
                    if sparql_query not in queries_set:
                        # file.write(sparql_query)
                        queries_set.add(sparql_query)
                        new_queries.append(sparql_query)

        if (object_uri, predicate_uri) not in object_predicate_set:
            object_predicate_set.add((object_uri, predicate_uri))
            # Handle range insertion
            if predicate_uri in predicate_range_mapping:
                range_identifier = predicate_range_mapping[predicate_uri].get("identifier", None)
                if range_identifier:
                    sparql_query = f"{object_uri} rdf:type <http://dice-research.org/{range_identifier}> .\n"
                    if sparql_query not in queries_set:
                        # file.write(sparql_query)
                        queries_set.add(sparql_query)
                        new_queries.append(sparql_query)

    # Batch write new queries to file
    with open(output_file, "a") as file:
        file.writelines(new_queries)






# def save_results_incrementally(results, output_file_path):
#     with open(output_file_path, 'a') as output_file:
#         json.dump(results, output_file, indent=4)

 # Save results incrementally to a file without duplicates.
def save_results_incrementally(results, output_file_path):
    try:
        # Load existing results
        with open(output_file_path, 'r') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"file not found error:{output_file_path} creating a new one")
        existing_results = []
    # Merge results while avoiding duplicates
    unique_results = {tuple(result.items()): result for result in existing_results + results}
    results_to_save = list(unique_results.values())
    with open(output_file_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)


def load_existing_results(output_file_path):
    if os.path.exists(output_file_path) and os.stat(output_file_path).st_size > 0:
        with open(output_file_path, 'r') as output_file:
            try:
                return json.load(output_file)
            except json.JSONDecodeError:
                print(f"Warning: {output_file_path} contains invalid JSON. Starting again neww")
                return []
    return []

# Collect subject and object URIs to query in batches
sub_uris_to_query = set()
global uris_to_query
obj_uris_to_query = set()
global uris_to_query
# to handle empty known predicates and unknown predicates
def process_unknown_predicate(triple, predicate_uri, output_file_path, seen_results, all_results):
    s_uri, p_uri, o_uri = extract_triple_parts(triple)
    subject_uri = s_uri
    object_uri = o_uri
    domain_uri = None
    range_uri = None
    # Querying classes for subject and object
    if subject_uri not in entities_class_hirarchy.keys():
        sub_uris_to_query.add(s_uri)
        # subject_classes = query_class_hierarchy(subject_uri)
        # entities_class_hirarchy[subject_uri] = subject_classes
    else:
        subject_classes = entities_class_hirarchy[subject_uri]
        domain_ids = assign_single_identifier(predicate_uri, subject_classes, is_domain=True)
        domain_uri = f"http://dice-research.org/{str(domain_ids).replace('http://dice-research.org/', '')}"

    if object_uri not in entities_class_hirarchy.keys():
        obj_uris_to_query.add(o_uri)
        # object_classes = query_class_hierarchy(object_uri)
        # entities_class_hirarchy[object_uri] = object_classes
    else:
        object_classes = entities_class_hirarchy[object_uri]
        range_ids = assign_single_identifier(predicate_uri, object_classes, is_domain=False)
        range_uri = f"http://dice-research.org/{str(range_ids).replace('http://dice-research.org/', '')}"


    # Check if this result has already been added
    # result_tuple = (predicate_uri, domain_uri, range_uri)
    # save_result(
    #     {"predicate": predicate_uri, "domain": domain_uri, "range": range_uri},
    #     result_tuple, seen_results, all_results, output_file_path)

    # Perform batch querying if needed
    if len(sub_uris_to_query) >= 5000 or len(obj_uris_to_query) >= 5000:
        domain_uri, range_uri = batch_query_and_update_hierarchy(predicate_uri)

    # Save result if domain and range URIs are valid
    if domain_uri or range_uri:
        if domain_uri==None:
            domain_uri = range_uri.replace("RP", "DP")
        elif range_uri==None:
            range_uri = domain_uri.replace("DP", "RP")
        result_tuple = (predicate_uri, domain_uri, range_uri)
        save_result({"predicate": predicate_uri, "domain": domain_uri, "range": range_uri},
                    result_tuple, seen_results, all_results, output_file_path)



def batch_query_and_update_hierarchy(predicate_uri):
    global sub_uris_to_query, obj_uris_to_query
    domain_uri, range_uri = None, None
    # Batch query for subjects and update hierarchy
    if sub_uris_to_query:
        subject_classes_dict = batch_query_class_hierarchy(sub_uris_to_query)
        entities_class_hirarchy.update(subject_classes_dict)
        subject_classes = set()
        for key in subject_classes_dict.keys():
            subject_classes.update(subject_classes_dict[key])
        sub_uris_to_query.clear()
        domain_ids = assign_single_identifier(predicate_uri, subject_classes, is_domain=True)
        domain_uri = f"http://dice-research.org/{str(domain_ids).replace('http://dice-research.org/', '')}"

    # Batch query for objects and update hierarchy
    if obj_uris_to_query:
        object_classes_dict = batch_query_class_hierarchy(obj_uris_to_query)
        entities_class_hirarchy.update(object_classes_dict)
        object_classes = set()
        for key in object_classes_dict.keys():
            object_classes.update(object_classes_dict[key])
        obj_uris_to_query.clear()
        range_ids = assign_single_identifier(predicate_uri, object_classes, is_domain=False)
        range_uri = f"http://dice-research.org/{str(range_ids).replace('http://dice-research.org/', '')}"

    return domain_uri, range_uri
    # if result_tuple not in seen_results:
    #     all_results.append(predicate_results)
    #     seen_results.add(result_tuple)
    #     save_results_incrementally(all_results, output_file_path)
    #
    # else:
    #     print(f"Duplicate result for unknown predicate {predicate_uri} skipped.")



def process_triples(output_file_path, sparql_queries_output_file,triples_json_file):
    all_results = load_existing_results(output_file_path)
    # Creating a set to track duplicate results and prevent them from being added to the output file
    seen_results = set((r["predicate"], r["domain"], r["range"]) for r in all_results)

    try:
        triples_data = load_triples_from_json(triples_json_file)
        known_predicates, unknown_predicates = determine_known_and_unknown_predicates(triples_data)
        processed_triples = load_processed_triples()

        for predicate_uri in known_predicates:
            for triple in triples_data:
                if isinstance(triple, dict):
                    p_uri = triple['predicate_uri']
                    s_uri = triple['subject_uri']
                    o_uri = triple['object_uri']
                else:
                    p_uri = triple[1]
                    s_uri = triple[0]
                    o_uri = triple[2]
                if p_uri == predicate_uri:
                    triple_key = (s_uri, predicate_uri, o_uri)

                    # Skip duplicate triples
                    if triple_key in processed_triples:
                        print(f"Triple {triple_key} already processed. Skipping.")
                        continue

                    domain = get_domain(predicate_uri)
                    ranges = get_range(predicate_uri)

                    # assign DPX or RPX when domain or range has multiple classes
                    if len(domain) > 1:
                        domain_identifier = assign_single_identifier(predicate_uri, domain, is_domain=True)
                        domain_uri = f"http://dice-research.org/{str(domain_identifier).replace('http://dice-research.org/', '')}"
                        # domain_uri = f"http://dice-research.org/{domain_identifier}"
                    elif len(domain) == 1:
                        domain_uri = domain[0]

                    if len(ranges) > 1:
                        range_identifier = assign_single_identifier(predicate_uri, ranges, is_domain=False)
                        range_uri = f"http://dice-research.org/{str(range_identifier).replace('http://dice-research.org/', '')}"
                        # range_uri = f"http://dice-research.org/{range_identifier}"
                    elif len(ranges) == 1:
                        range_uri = ranges[0]

                    # Prevent empty domain or range from being processed
                    if not domain or not ranges:
                        if is_property_symmetric(predicate_uri):
                            if domain:
                                range_uri = domain_uri
                            else:
                                domain_uri = range_uri
                            predicate_results = {
                                "predicate": predicate_uri,
                                "domain": domain_uri,
                                "range": range_uri
                            }

                            # Check if this result has already been added to the output file
                            result_tuple = (predicate_uri, domain_uri, range_uri)
                            if result_tuple not in seen_results:
                                all_results.append(predicate_results)
                                # json.dump(predicate_results, output_file)
                                save_results_incrementally(all_results, output_file_path)
                                seen_results.add(result_tuple)
                            else:
                                print(f"Duplicate result for {predicate_uri} skipped.")

                        else:
                            print(f"Predicate {predicate_uri} has empty domain or range. Treating as unknown.")
                            process_unknown_predicate(triple, predicate_uri, output_file_path, seen_results,
                                                      all_results)

                    else:
                        predicate_results = {
                            "predicate": predicate_uri,
                            "domain": domain_uri,
                            "range": range_uri
                        }

                        # Check if this result has already been added to the output file
                        result_tuple = (predicate_uri, domain_uri, range_uri)
                        if result_tuple not in seen_results:
                            all_results.append(predicate_results)
                            save_results_incrementally(all_results, output_file_path)
                            seen_results.add(result_tuple)
                        else:
                            print(f"Duplicate result for {predicate_uri} skipped.")

                    processed_triples.add(triple_key)

        # Handle unknown predicates
        for predicate_uri in unknown_predicates:
            print(f"starting for unknown predicate {predicate_uri}")
            for triple in triples_data:
                if isinstance(triple, dict):
                    p_uri = triple['predicate_uri']
                    s_uri = triple['subject_uri']
                    o_uri = triple['object_uri']
                else:
                    p_uri = triple[1]
                    s_uri = triple[0]
                    o_uri = triple[2]
                if p_uri == predicate_uri:
                    triple_key = (s_uri, predicate_uri, o_uri)

                    # Skip duplicate triples
                    if triple_key in processed_triples:
                        # print(f"Triple {triple_key} already processed. Skipping.")
                        continue

                    process_unknown_predicate(triple, predicate_uri, output_file_path, seen_results, all_results)
                    processed_triples.add(triple_key)
        # save_results_incrementally(all_results, output_file_path)
    finally:
        # print("saveeeeeeeeeeeeeeddd")
        save_processed_triples(processed_triples)
        class_mapping()
        save_results_incrementally(all_results, output_file_path)
        save_sparql_queries(sparql_queries_output_file, triples_data)
        print("Results saved")
import sys

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
def is_property_symmetric(property_id):
    """
    Check if a Wikidata property has a symmetric constraint.

    :param property_id: The ID of the Wikidata property (e.g., "P20").
    :return: True if the property has a symmetric constraint, False otherwise.
    """
    endpoint_url = "https://query.wikidata.org/sparql"
    # HTTP headers
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    # SPARQL ASK query
    # ask_query = f"""
    # ASK {{
    #   {property_id} p:P2302 ?statement .
    #   ?statement ps:P2302 wd:Q21510862 .
    # }}
    # """
    query = f"""PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            ASK {{
                wd:{property_id.split('/')[-1].replace(">","")}   p:P2302 ?statement .
                ?statement ps:P2302 wd:Q21510862 .
            }}"""


    print(query)
    results = get_results(endpoint_url, query)
    # for result in results["results"]["bindings"]:
    return results.get("boolean", False)


def read_triples(file_path):
    triples = set()

    with open(file_path, 'r') as file:
        for line in file:
            # Clean up and split the line into subject, predicate, object
            parts = line.strip().split(' ')
            if len(parts) == 4:
                subject, predicate, object, dot = parts
                # Store the triple as a tuple
                triples.add((subject, predicate, object))

    return triples

def extract_triple_parts(triple):
    return (triple['subject_uri'], triple['predicate_uri'], triple['object_uri']) if isinstance(triple, dict) else triple

def save_result(predicate_results, result_tuple, seen_results, all_results, output_file_path):
    if result_tuple not in seen_results:
        all_results.append(predicate_results)
        save_results_incrementally(all_results, output_file_path)
        seen_results.add(result_tuple)
    # else:
    #     print(f"Duplicate result for {result_tuple} skipped.")

def process_orignal_triples(output_file_path, triples_data, sparql_queries_output_file, processed_triples_output_file, predicate_class_mappings_file_path, distinct_predicates_files=None):
    if not triples_data:
        global entities_class_hirarchy
        # entities_class_hirarchy = load_dictionary("entites_class_hirarichy_dict.json")
        # save_class_hierarchy()
        load_class_hierarchy()
        print(len(entities_class_hirarchy))
        processed_predicates = load_processed_predicates()
        # save_class_hierarchy()


        output_file_base = output_file_path.replace(".json", "")
        processed_file_base = processed_triples_output_file.replace(".nt", "")
        sparql_queries_base = sparql_queries_output_file.replace(".txt", "")
        predicate_mappings_base = predicate_class_mappings_file_path.replace(".nt", "")

        for pred in distinct_predicates:
            if pred != 'http://www.wikidata.org/prop/direct/P166':
                continue

            print(f"extracting for predicate {pred}")
            if pred == 'http://www.wikidata.org/prop/direct/P20' or pred == 'http://www.wikidata.org/prop/direct/P19':
                continue

            if pred in processed_predicates:
                continue

            print(f"extracting for predicate {pred}")

            file_suffix = str(pred).split('/')[-1]
            output_file_path = f"{output_file_base}_{file_suffix}.json"
            processed_triples_output_file = f"{processed_file_base}_{file_suffix}.nt"
            sparql_queries_output_file = f"{sparql_queries_base}_{file_suffix}.txt"
            predicate_class_mappings_file_path = f"{predicate_mappings_base}_{file_suffix}.nt"

            triples_data = set(read_triples(f'predicate_wise_triples/{file_suffix}.nt'))
            print(f'length of triples: {len(triples_data)} for predicate {file_suffix}')
            if not triples_data:
                continue
            create_mapping_from_json_and_nt(output_file_path, predicate_class_mappings_file_path)

            if triples_data:
                all_results = load_existing_results(output_file_path)
                # Creating a set to track duplicate results and prevent them from being added to the output file
                seen_results = set((r["predicate"], r["domain"], r["range"]) for r in all_results)
                try:

                    known_predicates, unknown_predicates = determine_known_and_unknown_predicates(triples_data)
                    processed_triples = load_processed_triples(processed_triples_output_file)

                    # # Creating a set to track duplicate results and prevent them from being added to the output file
                    domains_dict, ranges_dict, semmetric_predicates = {}, {}, {}

                    with open(output_file_path, 'a') as output_file:
                        for predicate_uri in known_predicates:
                            # Process known predicates
                            for triple in triples_data:
                                s_uri, p_uri, o_uri = extract_triple_parts(triple)
                                if p_uri != predicate_uri:
                                    continue
                                if p_uri == predicate_uri:
                                    triple_key = (s_uri, predicate_uri, o_uri)

                                    # Skip duplicate triples
                                    if triple_key in processed_triples:
                                        print(f"Triple {triple_key} already processed. Skipping.")
                                        continue
                                    if predicate_uri not in domains_dict.keys():
                                        domain = get_domain(predicate_uri)
                                        domains_dict[predicate_uri] = domain
                                        # assign DPX or RPX when domain or range has multiple classes
                                        if len(domain) > 1:
                                            domain_identifier = assign_single_identifier(predicate_uri, domain,
                                                                                         is_domain=True)
                                            domain_uri = f"http://dice-research.org/{str(domain_identifier).replace('http://dice-research.org/', '')}"
                                            # domain_uri = f"http://dice-research.org/{domain_identifier}"
                                        elif len(domain) == 1:
                                            domain_uri = domain[0]
                                        else:
                                            domain_uri = None

                                    else:
                                        domain = domains_dict[predicate_uri]

                                    if predicate_uri not in ranges_dict.keys():
                                        ranges = get_range(predicate_uri)
                                        ranges_dict[predicate_uri] = ranges
                                        # assign DPX or RPX when domain or range has multiple classes
                                        if len(ranges) > 1:
                                            range_identifier = assign_single_identifier(predicate_uri, ranges,
                                                                                        is_domain=False)
                                            range_uri = f"http://dice-research.org/{str(range_identifier).replace('http://dice-research.org/', '')}"
                                            # range_uri = f"http://dice-research.org/{range_identifier}"

                                        elif len(ranges) == 1:
                                            range_uri = ranges[0]
                                        else:
                                            range_uri = None
                                    else:
                                        ranges = ranges_dict[predicate_uri]


                                    # Prevent empty domain or range from being processed
                                    if not domain or not ranges:
                                        if predicate_uri not in semmetric_predicates.keys():
                                            semmetric_predicates[predicate_uri] = is_property_symmetric(predicate_uri)

                                        if semmetric_predicates[predicate_uri]:
                                            domain_uri, range_uri = domain_uri or range_uri, range_uri or domain_uri
                                            if domain_uri:
                                                range_uri = domain_uri
                                            elif range_uri:
                                                domain_uri = range_uri
                                            else:
                                                print(f"Predicate {predicate_uri} has empty domain and range. Treating as unknown.")
                                                unknown_predicates.add(predicate_uri)
                                                break



                                            # Check if this result has already been added to the output file
                                            result_tuple = (predicate_uri, domain_uri, range_uri)
                                            save_result(
                                                {"predicate": predicate_uri, "domain": domain_uri, "range": range_uri},
                                                result_tuple, seen_results, all_results, output_file_path)

                                        else:
                                            print(f"Predicate {predicate_uri} has empty domain or range. Treating as unknown.")
                                            unknown_predicates.add(predicate_uri)
                                            break
                                            # process_unknown_predicate(triple, predicate_uri, output_file_path, seen_results,
                                            #                           all_results)

                                    else:

                                        # Check if this result has already been added to the output file
                                        result_tuple = (predicate_uri, domain_uri, range_uri)
                                        save_result(
                                            {"predicate": predicate_uri, "domain": domain_uri, "range": range_uri},
                                            result_tuple, seen_results, all_results, output_file_path)

                                    processed_triples.add(triple_key)
                        print("unkown")
                        # Handle unknown predicates
                        # Group triples by predicate for faster lookup
                        predicate_to_triples = defaultdict(list)
                        for triple in triples_data:
                            s_uri, p_uri, o_uri = extract_triple_parts(triple)
                            predicate_to_triples[p_uri].append((s_uri, p_uri, o_uri))
                        # Process unknown predicates
                        for predicate_uri in unknown_predicates:
                            print(f"Processing unknown predicate {predicate_uri}")

                            if predicate_uri not in predicate_to_triples:
                                continue  # Skip if predicate not in triples

                            for s_uri, p_uri, o_uri in predicate_to_triples[predicate_uri]:
                                triple_key = (s_uri, predicate_uri, o_uri)

                                # Skip duplicate triples
                                if triple_key in processed_triples:
                                    continue

                                process_unknown_predicate(
                                    (s_uri, p_uri, o_uri), predicate_uri, output_file_path, seen_results, all_results
                                )
                                processed_triples.add(triple_key)
                    save_processed_predicate(pred)
                finally:
                    if len(sub_uris_to_query) > 0 or len(obj_uris_to_query) > 0:
                        domain_uri, range_uri = batch_query_and_update_hierarchy(predicate_uri)
                    save_class_hierarchy()
                    # save_dictionary("entites_class_hirarichy_dict.json")
                    save_results_incrementally(all_results, output_file_path)
                    save_processed_triples(processed_triples, processed_triples_output_file)
                    class_mapping(predicate_class_mappings_file_path=predicate_class_mappings_file_path)
                    save_sparql_queries(sparql_queries_output_file, triples_data, pred="<" + pred + ">")
                    # Perform batch querying if needed

                    print("Results saved")
                    # json.dump(all_results, output_file, indent=4)
                # except Exception as e:
                #     print(f"An error occurred: {e}")



# Function to save newly processed predicates to file
def save_processed_predicate(predicate, file_path="processed_predicates.txt"):
    ensure_file_exists(file_path)
    with open(file_path, "a") as file:
        file.write(predicate + "\n")

# Function to load processed predicates from file
def load_processed_predicates(file_path="processed_predicates.txt"):
    try:
        with open(file_path, "r") as file:
            return set(line.strip() for line in file)
    except FileNotFoundError:
        return set()



def get_triples_for_predicate(predicate, sparql_endpoint="http://0.0.0.0:9080/sparql"):
    """
    Retrieves all triples related to a specific predicate from a SPARQL endpoint.

    Args:
        predicate (str): The predicate URI to query.
        sparql_endpoint (str): The SPARQL endpoint URL.

    Returns:
        set: A set of triples in the format (subject, predicate, object).
    """
    # Initialize SPARQL client
    sparql = SPARQLWrapper(sparql_endpoint)

    # Define the SPARQL query
    query = f"""
    SELECT ?subject ?object
    WHERE {{
      ?subject <{predicate}> ?object .
    }}
    """
    print("Executing SPARQL query:", query)
    # Set the query and return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        # Execute the query
        results = sparql.query().convert()

        # Extract triples into a set
        triples = {
            (result["subject"]["value"], predicate, result["object"]["value"])
            for result in results["results"]["bindings"]
        }
        print(f"Returned results: {len(triples)} triples")
        return triples

    except Exception as e:
        print(f"An error occurred: {e}")
        return set()

def save_to_nt(triples, file_path):
    """
    Saves a list of triples to an .nt file in the N-Triples format.

    Args:
        triples (list of tuples): List of triples, where each triple is a tuple (subject, predicate, object).
        file_path (str): The path to the output .nt file.

    Returns:
        None
    """
    with open("predicate_wise_triples/"+file_path+".nt", 'w', encoding='utf-8') as file:
        for subject, predicate, obj in triples:
            # Format the triple as an N-Triples string
            triple = f"<{subject}> <{predicate}> <{obj}> .\n"
            file.write(triple)

def file_exists_and_has_data(file_path):
    """
    Checks if a file exists and has data.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists and is not empty, False otherwise.
    """
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0
original_data = True
if original_data:
    # filepath_orignal_data = '../reified_wikidata/BPDP_test_reified_wikidata_new.nt'
    # output_orignal_data_path = '../reified_wikidata/output_predicate_domains_ranges_orignal.jsonl'
    # sparql_queries_output_file = '../reified_wikidata/sparql_queries_orignal.txt'
    # processed_triples_output_file = '../reified_wikidata/processed_triples_orignal.nt'
    # predicate_class_mappings_file_path = '../reified_wikidata/predicate_class_mappings.nt'
    type = "train"
    filepath_orignal_data = '../favel_wikidata/wiki_output_'+type+'_factbench_final_final.ttl' #favel_'+type+'_factbench_wikidata.nt'  wiki_output_test_factbench_final
    output_orignal_data_path = '../favel_wikidata/FactBench_'+type+'_output_predicate_domains_ranges_new.json'

    sparql_queries_output_file = '../favel_wikidata/FactBench_'+'sparql_queries_'+type+'.txt'
    processed_triples_output_file = '../favel_wikidata/FactBench_'+'processed_triples_'+type+'.nt'
    predicate_class_mappings_file_path = '../favel_wikidata/FactBench_'+'predicate_class_mappings_'+type+'.nt'

    # filepath_orignal_data = '../favel_wikidata/wiki_output_' + type + '_factbench_final.ttl'  # favel_'+type+'_factbench_wikidata.nt'  wiki_output_test_factbench_final
    #     filepath_orignal_wikipedia_data = '../favel_wikidata/orignal_factbench_'+type+'.ttl'
    #     output_orignal_data_path = type+'_output_predicate_domains_ranges_new.json'
    #
    #     sparql_queries_output_file = 'sparql_queries_'+type+'.txt'
    #     processed_triples_output_file = 'processed_triples_'+type+'.nt'
    #     predicate_class_mappings_file_path = 'predicate_class_mappings_'+type+'.nt'

    output_file_path = ''
    # Create an RDF graph and parse the .nt file
    g = Graph()
    g.parse(filepath_orignal_data, format="nt")

    triples = []
    all_triples = set()
    distinct_predicates = set()
    # Extract and print information
    for stmt in g.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                           object=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")):
        triple = dict()
        truth_value = g.value(subject=stmt, predicate=URIRef("http://swc2017.aksw.org/hasTruthValue"))
        subject = g.value(subject=stmt, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"))
        predicate = g.value(subject=stmt, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"))
        object_ = g.value(subject=stmt, predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object"))
        distinct_predicates.add(str(predicate).replace("https:","http:").replace("http://www.wikidata.org/entity/","http://www.wikidata.org/prop/direct/"))
        # Append the triple as a dictionary to the list
        if subject == None or predicate == None or object_ == None:
            print("none triple")
            continue
        if float(truth_value) == 1.0:
            triples.append({
                'subject_uri': str(subject),
                'predicate_uri': str(predicate),
                'object_uri': str(object_)
            })
        else:
            print(f"false triple: {subject, predicate, object_}")
    for pred in distinct_predicates:
        path = str(pred).split('/')[-1]
        if not file_exists_and_has_data("predicate_wise_triples/"+path+".nt"):
            all_triples = get_triples_for_predicate(pred)
            save_to_nt(all_triples, path )

    process_orignal_triples(output_file_path=output_orignal_data_path, triples_data=set(),sparql_queries_output_file=sparql_queries_output_file, processed_triples_output_file=processed_triples_output_file,predicate_class_mappings_file_path=predicate_class_mappings_file_path,distinct_predicates_files=distinct_predicates)


else:
    output_filepath = 'test_output_predicate_domains_ranges.json'
    sparql_queries_output_file = 'test_sparql_queries.txt'
    triples_json_file = "fever_triples_data.json"
    predicate_set = set()
    process_triples(output_filepath, sparql_queries_output_file, triples_json_file)



