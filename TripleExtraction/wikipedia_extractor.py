import requests
import rdflib
import os
from bs4 import BeautifulSoup
from maverick.coref import resolve_coreferences


class WikipediaDataExtractor:
    def __init__(self, nt_file, output_dir):
        self.nt_file = nt_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.graph = rdflib.Graph()
        self.graph.parse(nt_file, format='nt')
        self.wiki_base = "https://en.wikipedia.org/wiki/"

    def fetch_wikipedia_content(self, entity):
        entity_title = entity.split('/')[-1]
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{entity_title}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            links = [a['href'] for a in soup.find_all('a', href=True) if 'wikipedia.org/wiki/' in a['href']]
            return text, links
        return None, []

    def process_triples(self):
        for s, p, o in self.graph:
            if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
                subject_text, subject_links = self.fetch_wikipedia_content(str(s))
                object_text, object_links = self.fetch_wikipedia_content(str(o))

                # Process linked Wikipedia pages (1-hop)
                for link in subject_links[:5]:  # Limit to first 5 links
                    linked_text, _ = self.fetch_wikipedia_content(link)
                    if linked_text:
                        subject_text += '\n' + linked_text

                for link in object_links[:5]:
                    linked_text, _ = self.fetch_wikipedia_content(link)
                    if linked_text:
                        object_text += '\n' + linked_text

                # Apply coreference resolution
                if subject_text:
                    subject_text = resolve_coreferences(subject_text)
                    self.save_content(str(s), subject_text)
                if object_text:
                    object_text = resolve_coreferences(object_text)
                    self.save_content(str(o), object_text)

    def save_content(self, entity, content):
        filename = os.path.join(self.output_dir, entity.split('/')[-1] + ".txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved: {filename}")

# Example usage:
# extractor = WikipediaDataExtractor("dataset.nt", "output_folder")
# extractor.process_triples()
