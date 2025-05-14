import json
import logging
# import ollama
# import requests
import os
import re
import subprocess
import requests
# from ollama import Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



model_id = "mistralai/Mixtral-8x7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# model = AutoModelForCausalLM.from_pretrained(model_id)


from nltk.tokenize import sent_tokenize
def tokenize_into_sentences(paragraph):
    sentences = sent_tokenize(paragraph)
    return sentences


# url: str = "http://tentris-ml.cs.upb.de:8000/api/generate"):

def is_contextual_match(string1, string_list, similarity_threshold=0.5):
    def is_abbreviation(s, word):
            # Checks if 's' is an abbreviation of 'word'
        it = iter(word)
        return all(char in it for char in s)

        # Calculate cosine similarity scores
        # Create a list containing the input string followed by the string list
    if not isinstance(string1,str):
        return False
    if isinstance(string_list,str)==True:
        string_list = [string_list]
    documents = [string1] + list(string_list)  # Ensure we are working with a list
    if len(documents)<2:
        return False
    # print("document"+str(documents))
    cosine_failed = False
    try:
        vectorizer = CountVectorizer().fit_transform(documents)
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
    except:
        cosine_failed = True

        # Check the similarity score of the string against each string in the list
    for i, candidate in enumerate(string_list):
            # Check for contextual match (substring match)
        if string1 in candidate:
            return True

            # Check if the string is an abbreviation of any string in the list
        if is_abbreviation(string1, candidate):
            return True

            # Check if the string is a subset (set of characters present in the list element)
        if set(string1).issubset(set(candidate)):
            return True

            # Check cosine similarity score
        if not cosine_failed and cosine_sim[0][i + 1] >= similarity_threshold:  # i + 1 because the first row is the input string
            return True

        # If no matches found, return False
    return False
# llama3:70b
# mixtral:8x7b
class LLMStanceDetector:
    def __init__(self,
                 entities: list,
                 claim: str, model: str = "mistral",
                 url: str = "http://localhost:11436/api/chat",
                 port: str="11434"):
        self.entities = entities
        # self.client = Client(host='http://localhost:'+port)
        self.model = model
        self.url = url
        # example1 = (f"Given claim is: The polar bear population has been growing. "
        #            f"Given entities are: Polar bear and Population "
        #             f" Output relationship is: growing and triple (Polar bear, growing,  Population, 1.0) "
        #             f" So the final JSON output should be like: \"entity1\": Polar bear, \"relation\": growing, \"entity2\":Population, \"relationship_strength\":1.0 ")


        # example2 = (f"Given claim is: Global warming is driving polar bears toward extinction"
        #            f"Sentences supporting the given claim in the given textual document are: Environmental impacts include the extinction or relocation of many species as their ecosystems change, most immediately the environments of coral reefs, mountains, and the Arctic. Rising global temperatures, caused by the greenhouse effect, contribute to habitat destruction, endangering various species, such as the polar bear. ")
        #
        # example3 = (f"Given claim is: the bushfires in Australia were caused by arsonists and a series of lightning strikes, not 'climate change'."
        #             f"Sentences failed to support or refute the given claim in the given textual document are:  The 2007 Kangaroo Island bushfires were a series of bushfires caused by lightning strikes on 6 December 2007 on Kangaroo Island, South Australia, resulting in the destruction of 95,000 hectares (230,000 acres) of national park and wilderness protection area. Many fires are as a result of either deliberate arson or carelessness, however, these fires normally happen in readily accessible areas and are rapidly brought under control. Man-made events include arcing from overhead power lines, arson, accidental ignition in the course of agricultural clearing, grinding and welding activities, campfires, cigarettes and dropped matches, sparks from machinery, and controlled burn escapes. The fires would have been caused by both natural phenomenon and human hands. A summer heat wave in Victoria, Australia, created conditions which fuelled the massive bushfires in 2009.")
        record_delimiter = ","
        # example = example.replace("\n", "")
        # self.prompt = ("[INST]You are a highly specialized Relation Extraction model designed to extract relationships between the given set of entities from textual data and a given set of entites as a subject and object. Your goal is to identify meaningful relationships between entities in the text and represent them as structured triples. You have the following characteristics:"
        #                 f" Precision-Focused: You prioritize accuracy in identifying and extracting relationships, ensuring that each subject, predicate, and object is contextually correct."
        #                 f" Knowledge of Entities and Relations: You have been trained to recognize and differentiate between various named entities (e.g., people, organizations, locations) and can classify relationships (e.g., actions, affiliations, attributions) from the input text, which contains news articles spanning various domains."
        #                 f" Context-Aware: You understand that the meaning of a sentence often depends on its context. You are able to analyze multi-sentence passages and consider cross-sentence relationships when extracting triples."
        #                 f" Adaptable to Various Structures: You can handle complex sentence structures, including passive voice, nested clauses, or implied relationships, and still accurately extract triples."
        #                 f" Bias-Free: You operate impartially, avoiding any pre-existing biases when analyzing content. You focus solely on extracting accurate and factual information, as presented in the input text."
        #                 f" Efficient and Organized: You organize extracted triples in a structured and machine-readable format (i.e., JSON), ensuring clarity and consistency in your output."
        #                 f" From the given entities as input, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other. For each pair of related entities extract relationship."
        #                 f" Examples of your tasks:\n"
        #                 f" ############################# \n"
        #                 f" Example 1:"
        #                 f" Input Sentence: \"Apple Inc. acquired Beats Electronics in 2014.\", and Input Entities: \"Apple Inc.\" and \"Beats Electronics\". "
        #                 f" Triple: (\"Apple Inc.\", \"acquired\", \"Beats Electronics\"). "
        #                 f" Example 2: "
        #                 f" Input Sentence: \"Elon Musk, the CEO of Tesla, announced a new AI initiative., and input entities: \"Elon Musk\" and \"new AI initiative\". "
        #                 f" Triple: (\"Elon Musk\", \"announced\", \"new AI initiative\"). "
        #                 f" Example 3: "
        #                 f" Input Sentence: \"Climate change is causing severe weather patterns, scientists claim.\", and input entities: \"Climate change\", \"severe weather patterns\", \"scientists\", and \"severe weather patterns are caused by climate change\". "
        #                 f" Triple: (\"Climate change\", \"is causing\", \"severe weather patterns\"), (\"scientists\", \"claim\", \"severe weather patterns are caused by climate change\"). "
        #                 f" Special Instructions: "
        #                 f" ONLY generate relationships between this list of entities: {str(entities)}. "
        #                 f" Do not generate entities or use pronouns as entities. Use exactly the same entity names as input. "
        #                 f" If a sentence contains multiple relationships between given entities, you should extract multiple triples.\" "
        #                 f" Ensure that each triple represents a full and clear relationship, avoiding incomplete or vague entries. "
        #                 f" Each triple should be a single JSON object with the following keys: \"subject\", \"predicate\", \"object\". Each triple should always contain these 3 keys."
        #                 f" Multiple triples should be in a single list of objects. "
        #                 f" Return output in a clean, structured format in JSON, making it easy to parse."
        #                 f" Must to be followed Instructions: "
        #                 f" Only infer relation between given entities, nothing else. "
        #                 f" Do not use any other entities as subject or object, except the provided Entities, i.e., {str(entities)} "
        #                 f" Only use {str(entities)} entities as subject or object in triple."
        #                 f" Output as should be in machine readable JSON format. [/INST]"
        #                 f" Real Data\n"
        #                 f" ######################\n"
        #                 f" Input Sentence: {claim} and Input Entities: {str(entities)}")

        self.cot_prompt = (
            "[INST] Your task is to extract knowledge triples from text."
            " A knowledge triple consists of three elements: (1) Subject, (2) Predicate, (3) Object."
            " Subjects and objects must be entities from the provided list, and the predicate represents the relation between them."
            " Think step by step before extracting the triples."

            "\n### Example ###"
            "\nText: Audi AG is a German automobile manufacturer that designs, engineers, produces, markets and distributes luxury vehicles."
            "\nEntities: [\"Audi\", \"German\", \"automobile manufacturer\", \"luxury vehicle\", \"Volkswagen Group\", \"Ingolstadt\", \"Bavaria\"]"

            "\nStep 1 - Identify entity types:"
            "\n - \"Automobile manufacturer\" is a concept."
            "\n - \"Luxury vehicle\" is a concept."

            "\nStep 2 - Identify potential relations:"
            "\n - Possible predicates: [\"country\", \"owned by\", \"headquarters location\"]"

            "\nStep 3 - Extract triples:"
                "\n ```json"
            "\n [{\"subject\": \"Audi\", \"predicate\": \"Country\", \"object\": \"Germany\"},"
            "\n  {\"subject\": \"Audi\", \"predicate\": \"Owned by\", \"object\": \"Volkswagen Group\"},"
            "\n  {\"subject\": \"Audi\", \"predicate\": \"Headquarters location\", \"object\": \"Ingolstadt\"}]"
            "\n ```"
            "\n### Example Ends ###"

            "\n### Your Task ###"
            f"\n Extract knowledge triples from the following text: {claim}"
            f"\n Use only these entities as subjects and objects: {str(entities)}"
            "\n Output must be a valid JSON list in machine-readable format."
            "\n Do not include explanations, reasoning, or additional text."
            "\n [/INST]"
        )

        # self.cot_prompt = ("[INST]Your task is extracting knowledge triples from text. "
        #                     f" A knowledge triple consists of three elements: subject - predicate - object. Subjects and objects are entities and the predicate is the relation between them."
        #                     f" Before extracting triples, let’s think step by step."
        #                     f" Here is an example: Text: Audi AG () is a German automobile manufacturer that designs, engineers, produces, markets and distributes luxury vehicles. Audi is a subsidiary of Volkswagen Group and has its roots at Ingolstadt, Bavaria, Germany. Audi vehicles are produced in nine production facilities worldwide."
        #                     f" Let’s extract the entities first. Here is the list of the entities in this text: \"example entities\"=[\"Audi\", \"German\", \"automobile manufacturer\", \"luxury vehicle\", \"Volkswagen Group\", \"Ingolstadt\", \"Bavaria\"]"
        #                     f" What do you know about the entities? [\"Automobile manufacturer is a/an concept.\", \"Luxury vehicle is a/an concept.\"]"
        #                     f" Now we think about the potential relations between these entities: \"example predicates\"=[\"country\", \"owned by\", \"headquarters location\"]"
        #                     f" Now we can extract the triples: [[\"subject\":\"Audi\", \"predicate\":\"Country\", \"object\":\"Germany\"], [\"subject\":\"Audi\", \"predicate\":\"Owned by\", \"object\":\"Volkswagen Group\"], [\"subject\":\"Audi\", \"predicate\":\"Headquarters location\", \"object\":\"Ingolstadt\"]]"
        #                     f" Example ends here."
        #                     f" Output as should be in machine readable JSON format."
        #                     f" Do not use any other entities as subject and object, except: {str(entities)}. "
        #                     f" You are strictly not allowed to use any other entities."
        #                     f" Extract the triples from the following text using provided entities and thinking step by step.  [/INST]"
        #                     f" Text: {claim} and Input Entities: {str(entities)}."
        #                     f" Your answer: ")

        # self.prompt = (f"<s> [INST] "
        # f" Goal "
        # f" Given a textual claim and a set of entities that are potentially relevant to this activity, identify all relationships among entities needed from the text in order to capture the information and ideas in the text."
        # f" Steps"
        # f" From the entities given as input, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other. For each pair of related entities, extract the following information:"
        # f" - source_entity: name of the source entity, as provided in input"
        # f" - target_entity: name of the target entity, as provided in input"
        # f" - relationship: a single verb that explains why you think the source entity and the target entity are related to each other"
        # f" - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity between 0 and 1"
        # f" Format each relationship as (entity1, relationship, entity2, relationship_strength )"
        # f" Return only a single JSON object with the following keys: \"entity1\", \"relation\", \"entity2\", and \"relationship_strength\".  Each key should be followed by a single string value."
        # f" When finished, output"
        # f" Examples: "
        # f" ############################# [/INST] "+ example1)

    def remove_thinking_part(self, response):
        """
        Removes LLM thinking parts (handles unclosed tags and some variations).
        """
        # Handles <think>...</think>, <reasoning>...</reasoning>, and unclosed tags
        cleaned = re.sub(
            r'(<think>|<reasoning>|<internal>).*?(</think>|</reasoning>|</internal>|$)',
            '',
            response,
            flags=re.DOTALL
        )
        return cleaned.strip()
    def get_response_from_api_call(self):
        """
        :param text: String representation of an OWL Class Expression
        """
        # Define the URL and data payload

        url = os.environ.get('OLLAMA_URL', default='http://localhost:11434/api/generate')
        model = os.environ.get('MODEL',default='llama3.3:70b')


        data = {
            "model": model,
            "prompt": self.cot_prompt,
            "stream": False,
            "keep_alive": -1,
            "temperature": 0
        }

        # Send the POST request with JSON data
        response = requests.post(url, json=data)
        # response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))

        # if response.status_code == 200:
        #     res =  response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
        # else:
        #     res = f"Error: {response.status_code}, {response.text}"
        # data = {
        #     "model": model,
        #     "prompt": self.cot_prompt,
        #     "stream": False,
        #     "keep_alive": -1,
        #     "temperature": 0
        # }
        #
        # # Send the POST request with JSON data
        # response = requests.post(url, json=data)
        if not response.content or response.status_code != 200:
            logging.error(f"API returned an empty response or error: {response.status_code} - {response.text}")
            return [{"head": "Unknown", "type": "not-related", "tail": "Unknown"}]

        try:
            res = response.content
            # try:
            #     json_str = json.loads(res.decode('utf-8'))  # Ensure proper decoding
            # except json.JSONDecodeError as e:
            #     logging.error(f"JSON decoding error: {e}, Response content: {res}")
            #     return [{"head": "Unknown", "type": "not-related", "tail": "Unknown"}]

            if str(res).startswith("b'"):
                json_str = json.loads(res)
                # if 'response' in json_str.keys():
                #     json_str = json_str['response']
                # Extract 'response' key if it exists
                if isinstance(json_str, dict) and "response" in json_str:
                    json_str = json_str["response"]
                    json_str = self.remove_thinking_part(json_str)
                json_str = str(json_str).replace("```json\\n", "").replace("```json","").replace("```","")
                json_str = json_str.replace("\\n", "")  # Remove unnecessary newlines
                if str(json_str).__contains__("]"):
                    json_str = json.loads(json_str)

                res = json_str
            else:
                if str(res).__contains__("]"):
                    res = json.loads(str(res['response']))

            # json_str = json.loads(json.loads(response.content)['response'])
        except json.JSONDecodeError as e:
            # Handle JSON decoding error
            logging.error(str(response.content).replace('\n',''))
            logging.error("JSON decoding error:", e)
            # logging.error("{\"head\": \"" + self.entities.pop() + "\", \"type\": \"not-related\", \"tail\": \"" + self.entities.pop() + "\"}")
            return "{\"head\": \"" + self.entities.pop() + "\", \"type\": \"not-related\", \"tail\": \"" + self.entities.pop() + "\"}"

        json_str = res
        if not isinstance(json_str,list):
            json_str = [json_str]

        triples = []
        logging.info("LLM returns output:"+str(json_str))
        for obj in json_str:
            # ent1 = self.entities.pop()
            # ent2 = self.entities.pop()
            if isinstance(obj,dict) and "subject" in obj.keys():
                if not is_contextual_match(obj["subject"],self.entities):
                    continue
                if isinstance(obj["subject"], str):
                    ent1 = obj["subject"]
            else:
                continue
            if isinstance(obj,dict) and "object" in obj.keys():
                if not is_contextual_match(obj["object"],self.entities):
                    continue
                if isinstance(obj["object"], str):
                    ent2 = obj["object"]
            else:
                continue

            if isinstance(obj,dict) and "predicate" in obj.keys() and obj["predicate"] != "null"  and obj.get("predicate") is not None:
                logging.info("The relation between entities:" + str(obj["predicate"]))
                logging.info("----------------------------------------LLM Ends----------------------------------------")
                # {'head': 'play on you', 'type': 'opposite of', 'tail': 'back'}
                if ent1 is not None and ent2 is not None:
                    triples.append("{\"head\": \"" + ent1 + "\", \"type\": \"" + obj[
                        "predicate"] + "\", \"tail\": \"" + ent2 + "\"}")
                else:
                    print(
                        f"Skipping triple due to None value: ent1={ent1}, predicate={obj.get('predicate')}, ent2={ent2}")

                # triples.append("{\"head\": \"" + ent1 + "\", \"type\": \"" + obj["predicate"] + "\", \"tail\": \"" + ent2 + "\"}")
            else:
                logging.info(obj)
                triples.append( "{\"head\": \"" + ent1 + "\", \"type\": \"not-related\", \"tail\": \"" + ent2 + "\"}")
        if len(triples) == 0:
            triples.append("{\"head\": \"" + self.entities.pop() + "\", \"type\": \"not-related\", \"tail\": \"" + self.entities.pop() + "\"}")
        return triples



    # def get_response_from_api_call_2(self):
    #     """
    #     :param text: String representation of an OWL Class Expression
    #     """
    #
    #     response = ollama.chat(model=self.model, messages=[
    #       {
    #         'role': 'user',
    #         'content': self.prompt,
    #       },
    #     ])
    #     return response['message']['content']
# demir api
#     def get_response_from_req_api_call(self):
#         """
#         :param text: String representation of an OWL Class Expression
#         """
#
#         response = self.client.chat(model='mistral', messages=[
#             {
#                 'role': 'user',
#                 'content': self.prompt,
#             },
#         ])

        #
        # response = requests.get(url=self.url,
        #                         headers={"accept": "application/json", "Content-Type": "application/json"},
        #                         json={"model": self.model, "prompt": self.prompt})
        # print(response['message']['content'])
        # return response['message']['content']


    def get_line_from_file(self, file_path, search_pattern):
        # Define the grep command to search for the pattern in the file
        grep_command = ["grep", "-m", "1", search_pattern, file_path]

        # Execute the grep command
        result = subprocess.run(grep_command, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Extract and return the output line
            output_lines = result.stdout.splitlines()
            if output_lines:
                return output_lines[0]
            else:
                return None
        else:
            # Print an error message if the command failed
            print("Error executing grep command:", result.stderr)
            return None
