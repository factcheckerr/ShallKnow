import configparser
import json
import logging
import os
import queue
# from fastcoref import FCoref flask-3.1.0 kg-gen==0.1.6 numpy-2.2.2 srsly-2.5.1 spacy-3.8.2 thinc-8.3.2 blis-0.9.0 thinc-9.1.1 pytorch-2.5.1 scikit-learn-1.6.1 rapidfuzz-3.12.2
import threading
import uuid

import jsonlines
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM
# from fastcoref import LingMessCoref
from transformers import AutoTokenizer
from kg_gen import KGGen
import LLM_query
from entity_linking_utils import ner_el_func, check_if_synonyms_in_sentence, get_triples, start_entity_linking, \
    get_processed_data, read_processed_claims
from extract_triples_per_folder import read_files_in_folder, store_results



kg = KGGen(
    model="ollama_chat/deepseek-r1:14b",
    temperature=0.0,
    api_key=None
)

config = configparser.ConfigParser()
config_file = 'configuration.ini'
config.read(config_file)
log_path = config.get('logging', 'logs_path', fallback='nebula.log')
log_level = os.environ.get('log_level', default='ERROR') #config.get('logging', 'log_level', fallback='INFO')
log_level = getattr(logging, log_level.upper(), logging.INFO)
logging.basicConfig(filename=log_path, level=log_level,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='w')
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)  # Make sure it matches your desired logging level

# Set a formatter to match the file format
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)

nltk_download_path = config.get('logging', 'nltk_download_path')


nltk.download('treebank', download_dir=nltk_download_path)
spacy.cli.download("xx_sent_ud_sm")
# Your Python program code here


data = []
model_basename = "model"
use_triton = False

# logging.set_verbosity(logging.CRITICAL)
class Processor:
    def __init__(self, cache_path):
        self.count_articles = 0
        self.llm_try = 0
        self.llm_success = 0
        self.rel_ner_success = 0
        os.makedirs(cache_path, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large", cache_dir=cache_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large", cache_dir=cache_path)
        self.gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 1,
        }

    def read_jsonl(self, file_path):
        """
        Read a JSONL file and return a list of dictionaries.
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except:
                    continue

            # data = [json.loads(line) for line in file]
        return data

    def tokenize_into_sentences(self, paragraph):
        sentences = sent_tokenize(paragraph)
        return sentences

    def read_file_chunk(self, start, end, input_file_path):
        return data[start:end]

    def write_to_output(self, chunk, output_file2):
        with jsonlines.open(output_file2, "a") as json_file:
            for line in chunk:
                json_file.write(line)

    def extract_triplets(self, text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        return triplets

    def replace_with_coref_text(self, coref_data, tag_info):
        for data1 in coref_data:
            if data1['article_id'] == tag_info['article_id']:
                if bool(tag_info['claim']) and list(tag_info['claim'].keys())[0] == data1['original_sentence']:
                    tag_info['claim'] = {data1['coref_sentence']: list(tag_info['claim'].values())}
                    return tag_info
        return tag_info

    def dump_malformed_json(self, json_str):
        # Specify the filename
        filename = 'malformed_json.txt'

        # Write the malformed JSON to a file
        with open(filename, 'a') as file:
            file.write(json_str + "\n")


    def convert_to_serializable(self, obj):
        if isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def extract_triples(self, long_text, data_id, error_file):
        if long_text:
            try:
                try:
                    graph = kg.generate(
                        input_data=long_text,
                        cluster=True,
                        context="Extract triples"
                    )
                except Exception as gen_e:
                    graph = kg.generate(
                        input_data=long_text,
                        chunk_size=5000,
                        cluster=False,
                        context="Extract triples"
                    )
                
                print("\nGenerated Graph Output:", graph)
                try:
                    clustered_graph = kg.cluster(graph, context="Guide clustering")
                except Exception as cluster_e:
                    print("Error during clustering:", cluster_e)
                    clustered_graph = graph

                print("\nClustered Graph Output:", clustered_graph)

                if not isinstance(clustered_graph, dict):
                    graph_data = clustered_graph.__dict__
                else:
                    graph_data = clustered_graph

                serializable_graph = self.convert_to_serializable(graph_data)
                triples = []
                for tpl in serializable_graph['relations']:
                    triples.append({'head': tpl[0], 'type': tpl[1], 'tail': tpl[2]})
                return triples

            except Exception as e:
                with open(error_file, 'a') as err_f:
                    err_f.write(f"Error extracting triples: {long_text}, ID: {data_id}, Error: {str(e)}\n")
                return None

    def process_chunk(self, start, end, input_file_path, args):
        chunk = self.read_file_chunk(start, end, input_file_path)
        dataset_name = args["--dataset_name"]
        # dataset_path = args["--dataset_path"]
        dataset_type = args["--dataset_type"]
        # output_file_name = f"{dataset_type}_{args['--output_file_name']}"
        # output_file_mapping = os.path.join(dataset_path, 'output', output_file_name)

        # Load coreference data if needed
        coref_include = args["--add_coreference_layer"]
        coref_data = None
        if coref_include:
            coref_file_path = args["--coref_file_path"]
            coref_data = self.read_jsonl(os.path.join(coref_file_path, f"{dataset_name}_10_{dataset_type}_map.jsonl"))

        self.count_articles = 0
        self.rel_ner_success = 0
        self.llm_success = 0
        self.llm_try = 0
        triple_claim_dict = []
        entities_dict, relations_dict, error_ent_dict = get_processed_data(args['--dataset_path'], args['--entities_dict'], args['--error_ent_dict'], args['--relations_dict'])
        # error_ent_dict = []
        count =  0
        for tag_info in chunk:
            logging.info(f"Processing chunk: start={start}, end={end}, current={count}")
            count+=1
            self.process_document(args, tag_info, triple_claim_dict, coref_data)
            # Process and write triples periodically
            if len(triple_claim_dict) > 10:
                with lock:
                    start_entity_linking(triple_claim_dict, args, entities_dict,error_ent_dict, relations_dict)
                    # self.write_to_output(triple_claim_dict, output_file_mapping)
                    logging.error(f"Processed claims: {self.count_articles}, RelationExtraction successes: {self.rel_ner_success}, LLM attempts: {self.llm_try}, LLM successes: {self.llm_success}")
                    triple_claim_dict.clear()  # Clear the list after saving
            else:
                logging.info("Saving after 10 triples. Now Triples extracted:"+str(len(triple_claim_dict)))
    def serialize_sets(self,obj):
        if isinstance(obj, set):
            return list(obj)

        return obj


    def process_single_claim_api(self, claim, args, entities_dict,error_ent_dict, relations_dict):
        # input_file = sys.argv[1]
        logging.info("----------------------------------------------------------------")
        claim1 = claim['text']

        triple_claim_dict = []
        random_id = uuid.uuid4()

        tag_info = {
            'claim': f'{claim1}',
            'original_claim': f'{claim1}',
            'id': f'{random_id}'
        }
        # entities_dict, relations_dict = get_processed_data(args['--dataset_path'],args['--entities_dict'], args['--relations_dict'])
        self.process_single_claim(args, tag_info, triple_claim_dict)
        triples, entities_dict, relations_dict, error_ent_dict  = start_entity_linking(triple_claim_dict, args, entities_dict,error_ent_dict, relations_dict, bulk_triples=False)

        tpls =[]
        for triple in triples:
            dict1 = {
                "triple": {
                    triple[1]
                },
                "subject": triple[2],
                "predicate": triple[3],
                "object": triple[4],
                "claim": triple[5]
            }
            tpls.append(str(dict1))
        if len(tpls)>0:
            logging.error("triples returned:"+str(len(tpls)))
        claim['extracted_triples'] = str(tpls)

    # Shared data
    # entities_dict = {}
    # relations_dict = {}
    # error_ent_dict = {}

    # Queue to share results between producer and consumer
    result_queue = queue.Queue()

    def process_document_task(self, paragraphs, args):
        """Producer: Process document and add results to the queue."""
        for para in paragraphs:
            id = list(para.keys())[0]
            claim1 = para[id]
            tag_info = {
                'claim': f'{claim1}',
                'original_claim': f'{claim1}',
                'id': f'{id}'
            }

            # Process the document
            triple_claim_dict = []
            self.process_document(args, tag_info, triple_claim_dict)

            # Add result to the queue for entity linking
            self.result_queue.put((id, triple_claim_dict))

        # Signal the consumer thread to stop
        self.result_queue.put(None)

    def start_entity_linking_task(self,args, files_path, entities_dict, error_ent_dict, relations_dict):
        """Consumer: Fetch results from the queue and perform entity linking."""
        while True:
            item = self.result_queue.get()
            if item is None:  # Stop signal
                break

            id, triple_claim_dict = item

            # Start entity linking
            triples, entities_dict_local, relations_dict_local, error_ent_dict_local = start_entity_linking(
                triple_claim_dict, args, entities_dict, error_ent_dict, relations_dict, bulk_triples=False
            )

            # Process triples
            tpls = []
            for triple in triples:
                dict1 = {
                    "triple": list({triple[1]}),
                    "subject": triple[2],
                    "predicate": triple[3],
                    "object": triple[4],
                    "claim": triple[5]
                }
                tpls.append(dict1)

            if len(tpls) > 0:
                logging.error("triples returned:" + str(len(tpls)))

                # Write results to files
                output_file = files_path + "output_triples.jsonl"
                processed_sentences_file_path = files_path + "filtered_processed_sentences.txt"
                processed_hashes_file_path = files_path + "processed_articles_hashes.txt"

                store_results(output_file, tpls, id, processed_sentences_file_path,processed_hashes_file_path)
            self.result_queue.task_done()

    def process_document_api(self, claim, args, entities_dict,error_ent_dict, relations_dict):
        # input_file = sys.argv[1]
        logging.info("----------------------------------------------------------------")
        claim1 = claim['text'].encode('utf-8').decode('unicode-escape')

        triple_claim_dict = []
        random_id = uuid.uuid4()

        # entities_dict, relations_dict, error_ent_dict = get_processed_data(args['--dataset_path'], args['--entities_dict'], args['--error_ent_dict'], args['--relations_dict'])




        # entities_dict, relations_dict = get_processed_data(args['--dataset_path'],args['--entities_dict'], args['--relations_dict'])
        if not claim1.startswith("folder:"):
            tag_info = {
                'claim': f'{claim1}',
                'original_claim': f'{claim1}',
                'id': f'{random_id}'
            }
            self.process_document(args, tag_info, triple_claim_dict)
            triples, entities_dict, relations_dict, error_ent_dict  = start_entity_linking(triple_claim_dict, args, entities_dict,error_ent_dict, relations_dict, bulk_triples=False)
            tpls =[]
            for triple in triples:
                dict1 = {
                    "triple": {
                        triple[1]
                    },
                    "subject": triple[2],
                    "predicate": triple[3],
                    "object": triple[4],
                    "claim": triple[5]
                }
                tpls.append(str(dict1))
            if len(tpls)>0:
                logging.error("triples returned:"+str(len(tpls)))
            claim['extracted_triples'] = str(tpls)
        else:
            print("folder")
            files_path = claim1.split("folder:")[1].split(claim1.split("/")[-1])[0]
            folder_name = claim1.split("/")[-1]
            paragraphs = read_files_in_folder(files_path, folder_name)
            # Start threads
            producer_thread = threading.Thread(target=self.process_document_task, args=(paragraphs, args))
            consumer_thread = threading.Thread(target=self.start_entity_linking_task, args=(args, files_path, entities_dict, error_ent_dict, relations_dict))

            producer_thread.start()
            consumer_thread.start()

            # Wait for threads to complete
            producer_thread.join()
            consumer_thread.join()
            # for para in paragraphs:




    def process_single_claim(self, args, tag_info, triple_claim_dict, coref_data=None):
        # chunk = read_file_chunk(start, end, input_file_path)
        # Check for target tag and retrieve original claim
        triple_id_dict = []
        target_tag = args['--text_tag']
        target_text = tag_info.get(target_tag)
        if not target_text:
            logging.error("No target tag specified, please specify the tag or dictionary key for the claim!")
            return  # Exit if no target text
        self.count_articles += 1
        if isinstance(target_text, dict):
            # Access a specific key within the dictionary
            original_claim = list(target_text.keys())[0]  # Replace 'specific_key' with the appropriate key
        elif isinstance(target_text,str) and 'original_claim' in tag_info.keys():
            original_claim = tag_info.get('original_claim')
        elif isinstance(target_text,str) and not args["--add_coreference_layer"]:
            original_claim = tag_info.get('claim')
            tag_info['original_claim']= original_claim

        # Replace with coref text if included
        if "--add_coreference_layer" in args.keys() and args["--add_coreference_layer"]:
            if coref_data == None:
                logging.error("If coref. layer is added. Please provide coref_data as well.")
                exit(1)
            tag_info = self.replace_with_coref_text(coref_data, tag_info)
            target_text = tag_info.get(args['--text_tag'])

        if isinstance(target_text, dict):
            # Access a specific key within the dictionary
            target_text = list(target_text.keys())[0]  # Replace 'specific_key' with the appropriate key
        else:
            target_text = target_text

        target_text_id = tag_info.get(args['--text_tag_id'])
        sentences = self.tokenize_into_sentences(target_text) if '--short_claims' in args.keys() and not args['--short_claims'] else [target_text]

        # Extract relations
        relations = []
        for sentence in sentences:
            if len(sentence.split()) < 2:
                continue
            logging.info(
                f"\n-------------------------------------------------------------------------------------------- ------------------------------------------------")
            logging.info(f"Processing sentence: {sentence}")
            logging.info(
                f"-----------------------------NER/EL starts---------------------------------------------------------------------------------------------------\n")

            # Tokenizer text
            model_inputs = self.tokenizer(sentence, max_length=256, padding=True, truncation=True, return_tensors='pt')
            generated_tokens = self.model.generate(
                model_inputs["input_ids"].to(self.model.device),
                attention_mask=model_inputs["attention_mask"].to(self.model.device),
                **self.gen_kwargs,
            )
            logging.info('length of model generated tokens:' + str(len(generated_tokens)))
            frwd_entities = set()
            logging.info(
                f"-----------------------------RELIK REL extraction starts--------------------------------------------------------------------------------------")
            relik_extracted_triplets = get_triples(sentence)
            extracted_triplets = relik_extracted_triplets
            logging.info('length of extracted triplets:' + str(len(extracted_triplets)))
            logging.info(
                f"-----------------------------RELIK REL extraction completed--Babelscape/rebel-large started--------maybe second step(Babelscape/rebel-large) unnecessary?-------------------------------------------------------------------------")
            # Decode predictions
            try:
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                extracted_triplets, fe = self.process_extracted_triplets(extracted_triplets, sentence,
                                                                    original_claim, target_text_id,
                                                                    tag_info, triple_id_dict,
                                                                    forwarded_entities=frwd_entities, decoded_preds=decoded_preds)
            except Exception as e:
                logging.error(f"Exception occured:{e}")


            if not relik_extracted_triplets:  # Fallback to LLM if no triplets found
                logging.info(f"-----------------------------NER/EL fails...Fallback to LLM starts here-----------------------------------------------------------------")
                tt = ner_el_func(sentence, frwd_entities)
                if len(frwd_entities) < 2:
                    logging.info(
                        "< 2 Entities identified using NER system. Skipping LLM check for relation extraction. :-(")
                    continue
                self.llm_try += 1
                llm_extracted_triples = LLM_query.LLMStanceDetector(entities=frwd_entities,
                                                                    claim=sentence).get_response_from_api_call()
                extracted_triplets += self.process_llm_extracted_triples(llm_extracted_triples, original_claim,
                                                                    target_text_id, tag_info, sentence)
                logging.info(f"-----------------------------Fallback to LLM ends here-----------------------------------------------------------------------------------")

                if len(extracted_triplets) > 0:
                    self.llm_success += 1
            else:
                self.rel_ner_success += 1
            triple_claim_dict.extend(extracted_triplets)
        logging.info("\n\n\n")
        logging.info("--------------NER/REL finished, EL starts here. length of claims list:"+str(len(triple_claim_dict)))

    def process_document(self, args, tag_info, triple_claim_dict, coref_data=None):
        rabel_bool = os.environ.get('RABEL_INCLUDE', default='False')
        kg_gen_bool = os.environ.get('KG_GEN_INCLUDE', default='False')
        # chunk = read_file_chunk(start, end, input_file_path)
        # Check for target tag and retrieve original claim
        triple_id_dict = []
        target_tag = args['--text_tag']
        target_text = tag_info.get(target_tag)
        if not target_text:
            logging.error("No target tag specified, please specify the tag or dictionary key for the claim!")
            return  # Exit if no target text
        self.count_articles += 1
        if isinstance(target_text, dict):
            # Access a specific key within the dictionary
            original_claim = list(target_text.keys())[0]  # Replace 'specific_key' with the appropriate key
        elif isinstance(target_text,str) and 'original_claim' in tag_info.keys():
            original_claim = tag_info.get('original_claim')
        elif isinstance(target_text,str) and not args["--add_coreference_layer"]:
            original_claim = tag_info.get('claim')
            tag_info['original_claim']= original_claim

        # Replace with coref text if included
        if "--add_coreference_layer" in args.keys() and args["--add_coreference_layer"]:
            if coref_data == None:
                logging.error("If coref. layer is added. Please provide coref_data as well.")
                exit(1)
            tag_info = self.replace_with_coref_text(coref_data, tag_info)
            target_text = tag_info.get(args['--text_tag'])

        if isinstance(target_text, dict):
            # Access a specific key within the dictionary
            target_text = list(target_text.keys())[0]  # Replace 'specific_key' with the appropriate key
        else:
            target_text = target_text

        target_text_id = tag_info.get(args['--text_tag_id'])

        final_extracted_triples = dict()
        relik_extracted_triplets = dict()
        frwd_entities = dict()
        # Extract relations
        relations = []
        sentences = self.tokenize_into_sentences(target_text) if '--short_claims' in args.keys() and not args[
            '--short_claims'] else [target_text]
        if kg_gen_bool=='True':
            for sent in sentences:
                relik_extracted_triplets[sent] = self.extract_triples(sent,0,"error.txt")
                frwd_entities[sent] = set()
                extracted_triplets, fe = self.process_extracted_triplets(relik_extracted_triplets[sent], sent,
                                                                         original_claim, target_text_id,
                                                                         tag_info, triple_id_dict,
                                                                         forwarded_entities=frwd_entities[sent],
                                                                         decoded_preds=None)
                final_extracted_triples[sent] = extracted_triplets
        else:
            for sentence in sentences:
                if len(sentence.split()) < 2:
                    continue
                logging.info(
                    f"\n-------------------------------------------------------------------------------------------- ------------------------------------------------")
                logging.info(f"Processing sentence: {sentence}")
                logging.info(
                    f"-----------------------------NER/EL starts---------------------------------------------------------------------------------------------------\n")
                if rabel_bool=='True':
                    # Tokenizer text
                    model_inputs = self.tokenizer(sentence, max_length=256, padding=True, truncation=True, return_tensors='pt')
                    generated_tokens = self.model.generate(
                        model_inputs["input_ids"].to(self.model.device),
                        attention_mask=model_inputs["attention_mask"].to(self.model.device),
                        **self.gen_kwargs,
                    )
                    logging.info('length of model generated tokens:' + str(len(generated_tokens)))
                frwd_entities[sentence] = set()
                logging.info(
                    f"-----------------------------RELIK REL extraction starts--------------------------------------------------------------------------------------")

                relik_extracted_triplets[sentence] = get_triples(sentence)
                extracted_triplets = relik_extracted_triplets[sentence]
                logging.info('length of extracted triplets:' + str(len(extracted_triplets)))
                logging.info(
                    f"-----------------------------RELIK REL extraction completed--Babelscape/rebel-large started--------maybe second step(Babelscape/rebel-large) unnecessary?-------------------------------------------------------------------------")
                logging.info(f"Babelscape/rebel-large: {rabel_bool}")
                # Decode predictions
                try:
                    if rabel_bool=='True':
                        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                        extracted_triplets, fe = self.process_extracted_triplets(extracted_triplets, sentence,
                                                                                 original_claim, target_text_id,
                                                                                 tag_info, triple_id_dict,
                                                                                 forwarded_entities=frwd_entities[sentence],
                                                                                 decoded_preds=decoded_preds)
                    else:
                        extracted_triplets, fe = self.process_extracted_triplets(extracted_triplets, sentence,
                                                                                 original_claim, target_text_id,
                                                                                 tag_info, triple_id_dict,
                                                                                 forwarded_entities=frwd_entities[sentence], decoded_preds=None)
                    final_extracted_triples[sentence] = extracted_triplets

                except Exception as e:
                    logging.error(f"Exception occured:{e}")

        for triplet_sent in relik_extracted_triplets.keys():
            triplet = relik_extracted_triplets[triplet_sent]
            # triplet = []
            if len(triplet) >= 0:  # Fallback to LLM if no triplets found
                logging.info(
                    f"-----------------------------NER/EL fails...Fallback to LLM starts here-----------------------------------------------------------------")
                tt = ner_el_func(triplet_sent, frwd_entities[triplet_sent])
                if len(frwd_entities[triplet_sent]) < 2:
                    logging.info(
                        "< 2 Entities identified using NER system. Skipping LLM check for relation extraction. :-(")
                    continue
                self.llm_try += 1
                llm_extracted_triples = LLM_query.LLMStanceDetector(entities=frwd_entities[triplet_sent],
                                                                    claim=triplet_sent).get_response_from_api_call()
                extracted_triplets = self.process_llm_extracted_triples(llm_extracted_triples, original_claim,
                                                                        target_text_id, tag_info, triplet_sent)
                logging.info(
                    f"-----------------------------Fallback to LLM ends here-----------------------------------------------------------------------------------")
                if len(extracted_triplets) > 0:
                    final_extracted_triples[triplet_sent].extend(extracted_triplets)
                    self.llm_success += 1
                    # break  #kkkk
            else:
                self.rel_ner_success += 1

        for extracted_triplets in final_extracted_triples.values():
            if len(extracted_triplets) != 0:
                triple_claim_dict.extend(extracted_triplets)
        logging.info("\n\n\n")
        logging.info(
            "--------------NER/REL finished, EL starts here. length of claims list:" + str(len(triple_claim_dict)))

    def extract_original_claim(self, target_tag, tag_info):
        """Extract original claim from the tag info."""
        original_claim = tag_info.get(target_tag)
        if isinstance(original_claim, dict):
            return list(original_claim.keys())[0]
        return original_claim

    def process_extracted_triplets(self, triples, sentence, original_claim, target_text_id, tag_info,
                                   triple_id_dict, forwarded_entities, decoded_preds=None):
        """Process extracted triplets from model predictions."""
        # extracted_triplets = []
        triplets = triples
        toberemoved_triplets = []
        if decoded_preds == None:
            for tpl in triplets:
                if self.is_valid_triplet(tpl, sentence):
                    logging.info("saving dict.: --head:" + str(tpl.get('head')) + " --type " + str(
                        tpl.get('type')) + " --tail " + str(tpl.get('tail')))
                    forwarded_entities.add(tpl.get('head'))
                    forwarded_entities.add(tpl.get('tail'))
                    tpl["original_claim"] = original_claim
                    tpl["claim"] = sentence
                    tpl["id"] = target_text_id
                    # tpl["article_id"] = tag_info.get('article_id')
                    tpl["article_id"] = tag_info.get('article_id', tag_info.get('id'))
                    triple_id_dict.append(str(target_text_id))
                else:
                    toberemoved_triplets.append(tpl)
            for tpl in toberemoved_triplets:
                triplets.remove(tpl)
        else:
            for en_sentence in decoded_preds:
                trips = self.extract_triplets(en_sentence)
                for trip in trips:
                    if len(triplets) != 0:
                        triplets.append(trip) if trip not in triplets else None
                    else:
                        triplets = [trip] if not isinstance(trip, list) else trip
                toberemoved_triplets = []
                for tpl in triplets:
                    if self.is_valid_triplet(tpl, sentence):
                        logging.info("saving dict.: --head:" + str(tpl.get('head')) + " --type " + str(
                            tpl.get('type')) + " --tail " + str(tpl.get('tail')))
                        forwarded_entities.add(tpl.get('head'))
                        forwarded_entities.add(tpl.get('tail'))
                        tpl["original_claim"] = original_claim
                        tpl["claim"] = sentence
                        tpl["id"] = target_text_id
                        # tpl["article_id"] = tag_info.get('article_id')
                        tpl["article_id"] = tag_info.get('article_id', tag_info.get('id'))
                        triple_id_dict.append(str(target_text_id))
                    else:
                        toberemoved_triplets.append(tpl)
                for tpl in toberemoved_triplets:
                    triplets.remove(tpl)

        return triplets, forwarded_entities

    def is_valid_triplet(self, tpl, sentence):
        """Check if the triplet is valid based on its presence in the sentence."""

        # Iterate through each key-value pair in the dictionary `tpl`
        for key, value in tpl.items():

            # Skip keys that should not be validated
            if key not in {'head', 'tail'}:
                continue
            if len(value) == len(sentence):
                continue
            # Check if `value` is in the sentence or has a synonym in the sentence
            if not (value in sentence or check_if_synonyms_in_sentence(str(sentence.encode('utf-8').decode('unicode-escape')), value)):
                return False  # If any condition fails, return False

        # If all conditions pass, return True
        return True

    def process_llm_extracted_triples(self, llm_extracted_triples, original_claim, target_text_id, tag_info, sentence):
        """Process extracted triplets from the LLM."""
        extracted_triplets = []
        if isinstance(llm_extracted_triples,str):
            llm_extracted_triples = [llm_extracted_triples]
        for tpl in llm_extracted_triples:
            try:
                print(str(tpl))
                tpl = json.loads(tpl)
                if not isinstance(tpl,int) and tpl.get("type") != 'not-related' and tpl['tail']!='' and tpl['head']!='':
                    tpl["original_claim"] = original_claim
                    # just to see which claims are processed by
                    tpl["claim"] = "LLM output:-"+sentence
                    tpl["id"] = target_text_id
                    tpl["article_id"] = tag_info.get('article_id') if 'article_id' in tag_info else tag_info.get('id')
                    extracted_triplets.append(tpl)
            except json.JSONDecodeError as e:
                logging.error("JSON decoding error:", e)
                logging.error("data dumped:"+str(tpl))
                self.dump_malformed_json(tpl)

            # self.save_llm_processed(llm_extracted_triples)
        return extracted_triplets





class ArgumentParserMock:
    def __init__(self):
        self.args = {}

    def add_argument(self, name, default=None):
        self.args[name] = default

    def parse_args(self, args=[]):
        return self.args


# Create a lock
lock = threading.Lock()




def argparse_default(description=None):
    # Create an ArgumentParser instance
    parser = ArgumentParserMock()
    # TripleExtraction/rebel_output/NELA/split_files/output_1.jsonl
    parser.add_argument("--dataset_path", default='rebel_output/NELA/')
    parser.add_argument("--num_threads", default=1)

    # linking file names
    parser.add_argument("--output_IRIs_file", default='output_triples_IRIs.jsonl')
    parser.add_argument("--entities_dict", default='entities_dictionary.jsonl')
    parser.add_argument("--error_ent_dict", default='error_ent_dictionary.jsonl')
    parser.add_argument("--relations_dict", default='relations_dictionary.jsonl')
    
    parser.add_argument("--dataset_input_file_name", default='output_1.jsonl')  # for fever 'fever_paper_' + 'dataset_type(tain or test or dev)' + '.jsonl'
    # parser.add_argument("--output_file_name", default='output_triples_mapping.jsonl')  # for fever 'fever_paper_' + 'dataset_type(tain or test or dev)' + '.jsonl'
    parser.add_argument("--sentence_length_threshold", default=1)
    parser.add_argument("--text_tag", default='coreferenced_text')
    parser.add_argument("--text_tag_id", default='id')
    parser.add_argument("--short_claims", default=False)
    parser.add_argument("--dataset_type", default='val')
    parser.add_argument("--dataset_name", default='nela')

    parser.add_argument("--cache_path", default='/data/hub/cache')

    parser.add_argument("--add_coreference_layer", default=False)
    parser.add_argument("--coref_file_path", default='rebel_output/NELA/coref/')


    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)


def create_new_file_if_not_exist(file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as file:
            file.write('')
        logging.info(f"New file '{file_name}' created.")
    else:
        logging.info(f"File '{file_name}' already exists.")

def main():
    args = argparse_default()
    dataset = args["--dataset_path"]
    # dataset_name = args["--dataset_name"]
    # dataset = dataset + ''
    dataset_type = args["--dataset_type"]
    dataset_input_file_name = dataset_type + "_" + args["--dataset_input_file_name"]
    # output_file_name = args["--output_file_name"]

    text_tag = args["--text_tag"]

    path = args['--dataset_path']
    file_name = dataset_type +"_"+ args['--output_IRIs_file']  # '2-output_triples_IRIs.jsonl'

    # Define the file path
    output_file_path = path + 'output/output_'+dataset_type+'/' + file_name
    # exit(1)
    input_file_path = dataset + dataset_input_file_name
    # output_file_triples = dataset + 'output/output_' + dataset_type + '/output_triples.jsonl'
    # output_file_mapping = dataset + 'output/output_claimwise/output_' + dataset_type + '/'+output_file_name
    create_new_file_if_not_exist(output_file_path)
    # ID should be claim or ID
    claim_id_tag = 'id'  # or text_tag
    logging.info("start reading")

    # # Define a list to store the claims
    # path = args['--dataset_path']
    # dataset_type = args["--dataset_type"]
    # file_name = dataset_type + "_" + args['--output_IRIs_file']  # '2-output_triples_IRIs.jsonl'
    #
    # # Define the file path
    # output_file_path = path + 'linking/' + file_name

    # TODO: Check if file does not exist create it!!
    claims = list(set(read_processed_claims(output_file_path, tag='original_claim')))
    # claims = list(set(read_processed_claims(output_file_mapping, claim_id_tag)))
    logging.info("processed claims:" + str(len(claims)))

    i = 0

    # Reading the output file if already some data exists.
    with jsonlines.open(input_file_path, 'r') as file:
        for line in file:
            logging.info("checking processed claim:"+str(line[claim_id_tag]))
            if bool(line[text_tag]):
                val = line.get('article_id', line.get('id'))
                value = str(val) if isinstance(val, int) else val
                clm = list(line[text_tag].keys())[0] if isinstance(line[text_tag],dict) else line[text_tag]
                cc =value+'-'+clm
                if cc not in claims:
                    data.append(line)
                    i = i + 1
                else:
                    logging.info("claim already processed:"+cc)
            if i >100:
                break
    num_threads = args['--num_threads']  # Adjust the number of threads as needed

    # Get the size of the file
    file_size = len(data)
    print('file size: ' + str(file_size))
    # init_variables(args['--cache_path'])
    # processor = Processor(args['--cache_path'])
    # Calculate chunk size for each thread
    chunk_size = len(data) // num_threads
    print("chunksize:" + str(chunk_size))
    # exit(0)
    # Create threads
    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else file_size
        thread = threading.Thread(target=Processor(args['--cache_path']).process_chunk, args=(start, end, input_file_path, args))
        threads.append(thread)

    # Start threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()







