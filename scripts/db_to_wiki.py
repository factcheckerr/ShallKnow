import re
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

INPUT_FILE = "reified/BPDP_test_reified.nt"
OUTPUT_FILE = "BPDP_test_reified_wikidata_new.nt"
UNRESOLVED_LOG = "unresolved_resources_testt.log"

DBPEDIA_RESOURCE_REGEX = re.compile(r'^http://dbpedia\.org/resource/(.+)$')
DBPEDIA_ONTOLOGY_REGEX = re.compile(r'^http://dbpedia\.org/ontology/(.+)$')

DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# Cache dictionary to avoid redundant queries
mapping_cache = {}

# SPARQL query to get Wikidata IRI for a DBpedia resource
def get_wikidata_iri_via_sparql(dbpedia_resource_iri: str) -> str:
    if dbpedia_resource_iri in mapping_cache:
        return mapping_cache[dbpedia_resource_iri]

    sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)
    query = f"""
    SELECT ?sameAs WHERE {{
        <{dbpedia_resource_iri}> owl:sameAs ?sameAs .
        FILTER(STRSTARTS(STR(?sameAs), "http://www.wikidata.org/entity/"))
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            wikidata_iri = bindings[0]["sameAs"]["value"]
            mapping_cache[dbpedia_resource_iri] = wikidata_iri
            return wikidata_iri
    except Exception as e:
        print(f"SPARQL query failed for {dbpedia_resource_iri}: {e}")

    # Fallback to fuzzy matching if SPARQL fails
    return fallback_resolve_with_fuzzy_logic(dbpedia_resource_iri)

# SPARQL query to find the equivalent property for a DBpedia ontology property
def get_equivalent_property_via_sparql(dbpedia_ontology_iri: str) -> str:
    if dbpedia_ontology_iri in mapping_cache:
        return mapping_cache[dbpedia_ontology_iri]

    sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)
    query = f"""
    SELECT ?equivalentProperty WHERE {{
        <{dbpedia_ontology_iri}> owl:equivalentProperty ?equivalentProperty .
        FILTER(STRSTARTS(STR(?equivalentProperty), "http://www.wikidata.org/entity/"))
    }}
    LIMIT 1
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            equivalent_property = bindings[0]["equivalentProperty"]["value"]
            mapping_cache[dbpedia_ontology_iri] = equivalent_property
            return equivalent_property
    except Exception as e:
        print(f"SPARQL query failed for {dbpedia_ontology_iri}: {e}")

    # Fallback to fuzzy matching if SPARQL fails
    return fallback_resolve_property_with_fuzzy_logic(dbpedia_ontology_iri)

# Fuzzy search in Wikidata for unresolved resources
def fuzzy_search_wikidata(query: str, type_: str, language="en"):
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": language,
        "type": type_,
        "limit": 5,
        "format": "json"
    }
    try:
        response = requests.get(WIKIDATA_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("search", [])
    except Exception as e:
        print(f"Wikidata fuzzy search failed for query {query}: {e}")
    return []

# Fallback: Resolve unresolved resources with fuzzy logic
def fallback_resolve_with_fuzzy_logic(dbpedia_resource_iri: str) -> str:
    label = dbpedia_resource_iri.split("/")[-1]
    # Handle composite labels
    parts = label.split("__")
    refined_label = parts[0].replace("_", " ") if parts else label.replace("_", " ")
    matches = fuzzy_search_wikidata(refined_label, type_="item")
    if matches:
        best_match = matches[0]
        wikidata_iri = f"https://www.wikidata.org/entity/{best_match['id']}"
        mapping_cache[dbpedia_resource_iri] = wikidata_iri
        return wikidata_iri

    # Log unresolved resource
    log_unresolved("Resource", dbpedia_resource_iri, refined_label)
    return dbpedia_resource_iri

# Fallback: Resolve unresolved properties with fuzzy logic
def fallback_resolve_property_with_fuzzy_logic(dbpedia_ontology_iri: str) -> str:
    label = dbpedia_ontology_iri.split("/")[-1]
    refined_label = label.replace("_", " ")
    matches = fuzzy_search_wikidata(refined_label, type_="property")
    if matches:
        best_match = matches[0]
        wikidata_iri = f"https://www.wikidata.org/entity/{best_match['id']}"
        mapping_cache[dbpedia_ontology_iri] = wikidata_iri
        return wikidata_iri

    # Log unresolved property
    log_unresolved("Property", dbpedia_ontology_iri, refined_label)
    return dbpedia_ontology_iri

# Log unresolved entries for manual inspection
def log_unresolved(entity_type: str, dbpedia_iri: str, label: str):
    with open(UNRESOLVED_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"{entity_type}\t{dbpedia_iri}\t{label}\n")


# Replace DBpedia URIs with Wikidata IRIs or equivalent properties
def replace_if_dbpedia_uri(token: str) -> str:
    if token.startswith("<") and token.endswith(">"):
        inner_uri = token[1:-1]

        if DBPEDIA_RESOURCE_REGEX.match(inner_uri):
            new_uri = get_wikidata_iri_via_sparql(inner_uri)
            return f"<{new_uri}>"

        if DBPEDIA_ONTOLOGY_REGEX.match(inner_uri):
            new_uri = get_equivalent_property_via_sparql(inner_uri)
            return f"<{new_uri}>"

    return token

# Process a single line of the RDF file
def process_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith("#") or not line.endswith(" ."):
        return line

    triple_part = line[:-2].strip()
    triple_regex = re.compile(r'^(<[^>]+>|\".*?\")\s+(<[^>]+>|\".*?\")\s+(<[^>]+>|\".*?\")$')
    match = triple_regex.match(triple_part)
    if not match:
        return line

    subject_str, predicate_str, object_str = match.groups()
    subject_str = replace_if_dbpedia_uri(subject_str)
    predicate_str = replace_if_dbpedia_uri(predicate_str)
    object_str = replace_if_dbpedia_uri(object_str)

    return f"{subject_str} {predicate_str} {object_str} ."

# Main function to process the RDF file
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            new_line = process_line(line)
            fout.write(new_line + "\n")

    print(f"Processing complete. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
