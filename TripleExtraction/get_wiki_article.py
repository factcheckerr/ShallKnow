import json
import time
import os
import xml.etree.ElementTree as ET
import logging
import re
import html
import mwparserfromhell
from fastcoref import spacy_component
import spacy
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(filename='bpdp_error.txt', level=logging.ERROR, format='%(asctime)s %(message)s')

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

incremental_id = 1


# Step 1: load BPDP data
def clean_object_value(object_part):
    if '^^' in object_part:
        object_value = object_part.split('^^')[0].strip('"')
        return object_value
    else:
        object_value = object_part.strip('<>').rstrip(" ").rstrip('>')
        return object_value


def load_bpdp_data(bpdp_file):
    print("Loading RDF data")
    bpdp_data = []
    with open(bpdp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(' ', 2)
            if len(parts) == 3:
                subject_uri = parts[0].strip('<>')
                predicate_uri = parts[1].strip('<>')
                object_part = parts[2].rstrip('.').strip('<>').rstrip(" ").rstrip('>')

                cleaned_object_value = clean_object_value(object_part)

                bpdp_data.append((subject_uri, predicate_uri, cleaned_object_value))
    return bpdp_data


# Step 2: Filter relevant pages
def get_wikipedia_url_from_dbpedia_uri(dbpedia_uri):
    query = f"""
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?wikipediaURL
    WHERE {{
      <{dbpedia_uri}> foaf:isPrimaryTopicOf ?wikipediaURL .
      BIND(IRI(CONCAT("https://en.wikipedia.org/wiki/", STR(?wikipediaURL))) AS ?wikipediaURLFull)
      FILTER(STRSTARTS(STR(?wikipediaURLFull), "https://en.wikipedia.org/"))
    }}
    """
    endpoint = "https://dbpedia.data.dice-research.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            return result["wikipediaURL"]["value"]
    except Exception as e:
        print(f"Error fetching Wikipedia URL: {e}")
        return None


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def filter_relevant_pages(bpdp_data, relevant_url_file):
    """filter relevant Wikipedia pages from BPDP dataset."""
    print("Filtering relevant pages...")
    ensure_directory_exists(relevant_url_file)

    relevant_uris = {}
    if os.path.exists(relevant_url_file):
        with open(relevant_url_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                relevant_uris[entry['uri']] = entry['page']

    pages_list = []
    uri_list = set()

    for subject_uri, predicate_uri, object_uri in bpdp_data:
        for uri in (subject_uri, object_uri):
            if uri.startswith("http://dbpedia.org/resource/") and uri not in relevant_uris:
                page_url = get_wikipedia_url_from_dbpedia_uri(uri)
                if page_url:
                    pages_list.append(page_url)
                    uri_list.add(uri)
                    relevant_uris[uri] = page_url
                    with open(relevant_url_file, 'a') as f_out:
                        f_out.write(json.dumps({'uri': uri, 'page': page_url}) + "\n")

    return pages_list


# Step 3: collect original Wikipedia titles and save to file
def collect_page_titles(relevant_pages_file):
    wikipedia_titles = set()
    if os.path.exists(relevant_pages_file):
        with open(relevant_pages_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                page = entry['page']
                page_title = get_page_title_from_url(page)
                wikipedia_titles.add(page_title.lower())
    return list(wikipedia_titles)

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title)

def get_page_title_from_url(wikipedia_url):
    return wikipedia_url.split('/')[-1].replace(' ', '_')

def save_titles_to_file(titles, output_file):
    with open(output_file, 'a') as f:
        for title in titles:
            f.write(f"{title}\n")
    print(f"Titles saved to {output_file}")

def load_titles_from_file(file_path):
    with open(file_path, 'r') as f:
        return {line.strip().lower() for line in f}

# Extract outgoing links from text
def extract_outgoing_links(text):
    links = re.findall(r'\[\[([^|\]:#]+)(?:\|[^\]]*)?\]\]', text)
    return {
        link.strip().replace(' ', '_').lower()
        for link in links
        if ':' not in link.strip()
    }

def clean_title(title):
    unwanted_prefixes = ["category:", "file:", "template:", "help:"]
    if any(title.startswith(prefix) for prefix in unwanted_prefixes):
        return None
    # Remove parentheses, punctuation, and spaces
    cleaned_title = re.sub(r'[\(\)\[\]\{\}<>:;.,!?\'"\\/*?]', "", title)
    cleaned_title = cleaned_title.replace(" ", "_")
    return cleaned_title if cleaned_title else None


# Step 4: load original titles and parse XML dump for outgoing links, saving them separately
def parse_wiki_dump_for_outgoing_links(input_file_path, original_titles, outgoing_titles_file):
    outgoing_links_set = set()
    try:
        context = ET.iterparse(input_file_path, events=('start', 'end'))
        _, root = next(context)

        for event, elem in context:
            if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                title_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
                text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')

                if title_elem is None or text_elem is None:
                    continue

                current_title = title_elem.text.lower()
                text_content = text_elem.text

                # Check if this page's title is in the original titles list
                if current_title in original_titles:
                    outgoing_links = extract_outgoing_links(text_content or "")
                    # Clean the links and filter out invalid ones
                    cleaned_links = {clean_title(link) for link in outgoing_links if clean_title(link)}
                    outgoing_links_set.update(cleaned_links)
                root.clear()

    except FileNotFoundError:
        print(f"Input file '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    # Save outgoing links to a separate file
    with open(outgoing_titles_file, 'a') as f:
        for link in sorted(outgoing_links_set):
            f.write(f"{link}\n")


def resolve_coreferences(text):
    try:
        doc = nlp(text, component_cfg={"fastcoref": {'resolve_text': True}})
        return doc._.resolved_text
    except Exception as e:
        logging.error(f"Coreference resolution failed: {e}")
        return text

def parse_redirect_from_text(text):
    redirect_match = re.search(r'#REDIRECT\s*\[\[\s*([^\]]+)\s*\]\]', text)
    if redirect_match:
        return redirect_match.group(1).strip()
    return None


def clean_text(wiki_text):
    """Cleans Wikipedia text while preserving paragraphs and removing unnecessary content."""

    # unescape HTML entities
    wiki_text = html.unescape(wiki_text)

    try:
        # parse with mwparserfromhell
        wikicode = mwparserfromhell.parse(wiki_text)

        # remove Templates (Infoboxes, References)
        for template in wikicode.filter_templates():
            wikicode.remove(template)

        # remove Comments
        for comment in wikicode.filter_comments():
            wikicode.remove(comment)

        # remove Magic Words (__TOC__, __NOINDEX__)
        for magic_word in wikicode.filter_tags(matches=lambda tag: "__" in tag):
            wikicode.remove(magic_word)

        # remove File & Image Links (e.g., [[File:Example.jpg]])
        for file in wikicode.filter_wikilinks():
            if file.title.lower().startswith(("file:", "image:")):
                wikicode.remove(file)

        # remove Category Links (e.g., [[Category:Science]])
        for category in wikicode.filter_wikilinks():
            if category.title.lower().startswith("category:"):
                wikicode.remove(category)

        # remove Table of Contents (TOC) and Footer Sections
        for section in wikicode.filter_headings():
            if any(word in section.title.lower() for word in
                   ["see also", "references", "external links", "further reading"]):
                section.extract()  # Remove the entire section

        cleaned_text = wikicode.strip_code(normalize=True)

    except Exception as e:
        logging.error(f"Error parsing wikitext with mwparserfromhell: {e}")
        cleaned_text = wiki_text

    # further remove extra spaces and clean special characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)  # Remove remaining HTML tags
    cleaned_text = re.sub(r'\[\[|\]\]', '', cleaned_text)  # Remove wiki link brackets
    cleaned_text = re.sub(r"'{2,5}", '', cleaned_text)  # Remove bold/italic formatting
    cleaned_text = re.sub(r'\(\s*;\s*\)', '', cleaned_text)  # Remove empty parentheses
    cleaned_text = re.sub(r'{[^{}]*}', '', cleaned_text)  # Remove stray braces
    cleaned_text = re.sub(r'\[\[Category:.*?\]\]', '', cleaned_text)  # Remove category tags
    cleaned_text = re.sub(r'\[\[File:.*?\]\]', '', cleaned_text)  # Remove file links
    cleaned_text = re.sub(r'\[\[Image:.*?\]\]', '', cleaned_text)  # Remove image links
    cleaned_text = re.sub(r'\[\[Special:.*?\]\]', '', cleaned_text)  # Remove special pages
    cleaned_text = re.sub(r'<ref.*?>.*?</ref>', '', cleaned_text, flags=re.DOTALL)  # Remove references
    cleaned_text = re.sub(r'\{\{.*?\}\}', '', cleaned_text)  # Remove remaining templates

    # preserve paragraphs
    cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)  # Keep newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Reduce excessive newlines

    return cleaned_text.strip()

def handle_redirects(text_content):
    """Handles Wikipedia redirects, returns target title if redirect exists."""
    redirect_match = re.match(r'#REDIRECT \[\[(.*?)\]\]', text_content, re.IGNORECASE)
    if redirect_match:
        redirected_title = redirect_match.group(1).strip()
        return clean_title(redirected_title)
    return None

def extract_and_save_page_content(input_file_path, titles_file, output_dir, error_log_file, checkpoint_file):
    global incremental_id

    with open(titles_file, 'r') as f:
        target_titles = set(line.strip().lower() for line in f)
        print(f"Total target titles: {len(target_titles)}")

    os.makedirs(output_dir, exist_ok=True)

    processed_titles = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as checkpoint:
            for line in checkpoint:
                if line.startswith("ID:"):
                    incremental_id = int(line.split(":")[1].strip())
                else:
                    processed_titles.add(line.strip())

    processed_count = 0
    new_redirected_target_titles = set()

    with open(error_log_file, 'a') as error_log, open(checkpoint_file, 'a') as checkpoint:
        try:
            context = ET.iterparse(input_file_path, events=('start', 'end'))
            _, root = next(context)

            for event, elem in context:
                if event == 'end' and elem.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
                    title_elem = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title')
                    text_elem = elem.find('.//{http://www.mediawiki.org/xml/export-0.10/}text')

                    if title_elem is None or text_elem is None:
                        continue

                    current_title = title_elem.text.lower()
                    if current_title not in target_titles or current_title in processed_titles:
                        root.clear()
                        continue

                    sanitized_title = sanitize_filename(current_title)
                    output_file_path = os.path.join(output_dir, f"{sanitized_title}.jsonl")
                    print(f"Processing '{sanitized_title}'")

                    try:
                        if text_elem.text:
                            # Handle redirects
                            redirected_title = handle_redirects(text_content)
                            if redirected_title:
                                target_titles.add(redirected_title)

                                # Save redirecttarget title
                                with open(final_titles_file, "a") as titles_out:
                                    titles_out.write(f"{redirected_title}\n")
                                # Skip processing, fetch and save the redirected title instead
                                continue

                            # Process content
                            #just to remove the footers and references
                            text_content = re.sub(r'== *(See also|References|External links|Further reading) *==.*', '',
                                                  text_content, flags=re.DOTALL)

                            original_text = clean_text(text_content)
                            coreference_text = resolve_coreferences(original_text)
                            page = {
                                'id': incremental_id,
                                'original_text': original_text,
                                'coreference_text': coreference_text
                            }
                            with open(output_file_path, 'a') as f_out:
                                f_out.write(json.dumps(page) + "\n")
                                print(f"Written content for '{current_title}' to {output_file_path}")

                            processed_titles.add(current_title)
                            checkpoint.write(f"ID:{incremental_id}\n")
                            checkpoint.write(f"{current_title}\n")
                            checkpoint.flush()
                            incremental_id += 1
                        else:
                            error_log.write(f"{current_title}\n")
                    except Exception as e:
                        logging.error(f"Error processing '{current_title}': {e}")
                    processed_count += 1

        except FileNotFoundError:
            print(f"Input file '{input_file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    print(f"Page contents saved in '{output_dir}', with errors logged to '{error_log_file}'.")

    return new_redirected_target_titles


if __name__ == '__main__':
    bpdp_file = 'reified/BPDP_train_reified.nt'
    relevant_pages_file = 'bpdp_combined_pages_urls.jsonl'
    original_titles_file = 'bpdp_original_titles.txt'
    xml_dump_file = '../enwiki-latest-pages-articles.xml'
    outgoing_titles_file = 'bpdp_outgoing_titles.txt'
    final_titles_file = 'bpdp_all_titles.txt'
    output_dir = 'bpdp_processed_wiki_pages'
    error_log_file = 'bpdp_content_fetch_error_titles.txt'
    checkpoint_file = 'bpdp_checkpoint.txt'

    # Step 1: load BPDP data
    bpdp_data = load_bpdp_data(bpdp_file)

    # Step 2: Filter relevant pages
    relevant_pages = filter_relevant_pages(bpdp_data, relevant_pages_file)
    print(relevant_pages)

    # Step 3: collect original Wikipedia titles and save to file
    wikipedia_titles = collect_page_titles(relevant_pages_file)
    save_titles_to_file(wikipedia_titles, original_titles_file)

    # Step 4: load original titles and parse XML dump for outgoing links, saving them separately
    original_titles = load_titles_from_file(original_titles_file)
    parse_wiki_dump_for_outgoing_links(xml_dump_file, original_titles, outgoing_titles_file)

    # Step 5: load both original and outgoing titles, combine, clean, sort, and save to final file
    all_titles = sorted(load_titles_from_file(original_titles_file).union(load_titles_from_file(outgoing_titles_file)))
    save_titles_to_file(all_titles, final_titles_file)

    # Step 5: extract the content of the page
    extract_and_save_page_content(xml_dump_file, final_titles_file, output_dir, error_log_file, checkpoint_file)

    # # Handle new redirects
    # while new_redirects:
    #     save_titles_to_file(new_redirects, final_titles_file)
    #     new_redirects = extract_and_save_page_content(xml_dump_file, final_titles_file, output_dir, error_log_file,
    #                                                   checkpoint_file)

