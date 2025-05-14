import configparser
from flask import Flask, jsonify, request
from threaded_extraction import Processor
import ast
import time
import logging
import common_utils as c_util
import json
import os
from entity_linking_utils import get_processed_data
import sys
from dotenv import load_dotenv

# Load .env only if running locally
if not os.environ.get('RUNNING_IN_DOCKER'):
    load_dotenv()

# logging.basicConfig(level=logging.INFO)
# configuration read
logging.info('Reading configuration file..')
config = configparser.ConfigParser()
config_file = 'configuration.ini'
config.read(config_file)
log_path = config.get('logging', 'logs_path', fallback='nebula.log')  # default to 'app.log' if not specified


input_file_path =  os.environ.get('dataset_path', default='/data/') #input_file_path = config.get('DEFAULT', 'dataset_path', fallback='/data/')
text_tag = config.get('DEFAULT', 'text_tag', fallback='/data/')
entities_dict_path = config.get('DEFAULT', 'entities_dict', fallback='entities_dictionary.jsonl')
error_ent_dict_path = config.get('DEFAULT', 'error_ent_dict', fallback='error_ent_dictionary.jsonl')
relations_dict_path = config.get('DEFAULT', 'relations_dict', fallback='relations_dictionary.jsonl')
output_IRIs_file = config.get('DEFAULT', 'output_IRIs_file', fallback='output_triples_IRIs.jsonl')
dataset_type = config.get('DEFAULT', 'dataset_type', fallback='single')
text_tag_id = config.get('DEFAULT', 'text_tag_id', fallback='id')

log_level = os.environ.get('log_level', default='ERROR') #config.get('logging', 'log_level', fallback='INFO')
# log_level = config.get('logging', 'log_level', fallback='INFO')
log_level = getattr(logging, log_level.upper(), logging.INFO)  # Convert log_level string to logging level
# Create log directory if it does not exist
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(filename=log_path, level=log_level,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='w')

cache_path = config.get('logging', 'cache_path')

entities_dict, relations_dict, error_ent_dict = get_processed_data(input_file_path, entities_dict_path, error_ent_dict_path, relations_dict_path)

request_counter = 0
app = Flask(__name__)
# input_file_path = sys.argv[1]
# if input_file_path == 'run':
#     input_file_path = '/data/nebula/TripleExtraction/NebulaTripleExtraction/rebel_output/NELA/claimwise_nela/'
def_placeholder = '00'
io_exc_list = ['query', 'full_json']

comp_inst_map = {}
comp_map = {
    'triple_extraction': Processor
}
path_pipeline_map = {}
def detect_components(config):
    """
    Function to detect required components in the configuration files.
    """

    for section in config:
        # Check if section is an Nebula Pipeline
        if section.strip().lower().startswith("nebula pipeline"):
            # extract pipeline name
            pipeline_name = config.get(section, 'name')
            # extract pipeline path
            pipeline_path = config.get(section, 'path')

            # extract pipeline components
            comp_list = json.loads(config.get(section, 'components'))
            logging.info("list of components to be loaded: %s" % str(comp_list))
            # find/add components in the instance map
            inst_list = []
            for comp in comp_list:
                if comp not in comp_inst_map:
                    comp_inst_map[comp] = comp_map[comp](cache_path)
                inst_list.append(comp_inst_map[comp])
            # map the pipeline path to pipeline name + pipeline instance list
            path_pipeline_map[pipeline_path] = {
                'name': pipeline_name,
                'comp_list': comp_list,
                'inst_list': inst_list
            }
    logging.info('Paths found:%s' % path_pipeline_map)

# Basic route to greet a user
@app.route('/bulk_dataset_process/<name>', methods=['GET'])
def greet(name):
    # Processor(cache_path).process_single_claim_api()
    return jsonify({"Need to be implemented this functionality at later stage if needed!!message": f"Hello, {name}!"})


# Route to add two numbers
# @app.route('/add', methods=['POST'])

@app.route('/dextract', methods=['POST'])
def doc_extract_triple():
    global request_counter
    request_counter += 1
    if request.form:
        data = request.form
    elif request.json:
        data = request.json
    logging.info('Query received at custom-pipeline: No.:' + str(request_counter))
    logging.info('Data received for extraction: %s' % data)
    comp_arr = data['components'].split(',')
    full_json = False
    doc_extract = True
    if ('full_json' in data) and data['full_json']:
        full_json = True
    inst_list = []
    for item in comp_arr:
        inst_list.append(comp_inst_map[item.strip()])

    if (len(inst_list) == len(comp_arr)) and ('query' in data):
        output = None
        data_q = data['query'].strip()
        try:
            if data_q.startswith('['):
                data_q = ast.literal_eval(data_q)
        except Exception as e:
            # do nothing
            pass
        # if the query is list, then process one at a time
        if type(data_q) == list:
            logging.debug('Processing query as a list.')
            output = []
            for query in data_q:
                output.append(process_query(query, data, inst_list, full_json, doc_extract))
        # else process the single query
        else:
            logging.debug('Processing a single query.')
            output = process_query(data_q, data, inst_list, full_json, doc_extract)
        return output
    else:
        return f'Invalid request'
@app.route('/extract', methods=['POST'])
def extract_triple():
    global request_counter
    request_counter += 1
    if request.form:
        data = request.form
    elif request.json:
        data = request.json
    logging.info('Query received at custom-pipeline: No.:'+str(request_counter))
    logging.info('Data received for extraction: %s' % data)
    comp_arr = data['components'].split(',')
    full_json = False
    doc_extract = False
    if ('full_json' in data) and data['full_json']:
        full_json = True
    inst_list = []
    for item in comp_arr:
        inst_list.append(comp_inst_map[item.strip()])

    if (len(inst_list) == len(comp_arr)) and ('query' in data):
        output = None
        data_q = data['query'].strip()
        try:
            if data_q.startswith('['):
                data_q = ast.literal_eval(data_q)
        except Exception as e:
            # do nothing
            pass
        # if the query is list, then process one at a time
        if type(data_q) == list:
            logging.debug('Processing query as a list.')
            output = []
            for query in data_q:
                output.append(process_query(query, data, inst_list, full_json,doc_extract))
        # else process the single query
        else:
            logging.debug('Processing a single query.')
            output = process_query(data_q, data, inst_list, full_json,doc_extract)
        return output
    else:
        return f'Invalid request'
    # data = request.get_json()
    # num1 = data.get('num1')
    # num2 = data.get('num2')
    # if num1 is None or num2 is None:
    #     return jsonify({"error": "Please provide both 'num1' and 'num2'"}), 400
    #
    # try:
    #     result = float(num1) + float(num2)
    #     return jsonify({"result": result})
    # except ValueError:
    #     return jsonify({"error": "Invalid input. Please provide numbers."}), 400

# Init config

# config = configparser.ConfigParser()
# config.read(config_file)
# Initialize the requested components
detect_components(config)

TOKEN_LIMIT = int(config['DEFAULT'].get('token_limit', '400'))

def process_query(query, data, inst_list, full_json, doc_extract):
    # Check if query exceeds the default token length limit
    query_tokens = c_util.tokenize_query(query)
    if len(query_tokens) < TOKEN_LIMIT or doc_extract:
        # proceed normally
        return process_normal_query(query, data, inst_list, full_json,doc_extract)
    else:
        logging.info('Query length %d exceeds the default limit %d. Splitting query into smaller chunks.' % (len(query_tokens), TOKEN_LIMIT))
        # divide query into sentences
        query_sentences = c_util.split_sentences(query)
        # Form chunks of sentences combined together smaller than default token limit
        query_chunks = []
        cur_chunk = ''
        cur_chunk_len = 0
        for sentence in query_sentences:
            sentence_tokens = c_util.tokenize_query(sentence)
            # if sentence itself is bigger than token limit then create it's own chunk
            if len(sentence_tokens) > TOKEN_LIMIT:
                # flush the previous chunk to query chunks
                if cur_chunk_len > 0:
                    query_chunks.append(cur_chunk.strip())
                # add current sentence
                query_chunks.append(sentence)
                # reset current chunk
                cur_chunk = ''
                cur_chunk_len = 0
            elif len(sentence_tokens) + cur_chunk_len <= TOKEN_LIMIT:
                # add to current chunk with a whitespace
                cur_chunk += ' ' + sentence
                cur_chunk_len += len(sentence_tokens)
                # continue loop to avoid resetting cur_chunk
                continue
            else:
                query_chunks.append(cur_chunk.strip())
                # reset current chunk
                cur_chunk = sentence
                cur_chunk_len = len(sentence_tokens)
        # flush remaining chunk
        if cur_chunk_len > 0:
            query_chunks.append(cur_chunk.strip())
        # logging
        logging.info('Total %d chunks formed: %s ' % (len(query_chunks), query_chunks))
        # loop through the chunks and process them as normal query
        results = []
        for chunk in query_chunks:
            ret_val = process_normal_query(chunk, data, inst_list, full_json, doc_extract)
            # If the retured json is empty, something went wrong, hence stop processing and return the same
            if len(ret_val) == 0 and isinstance(ret_val, dict):
                return ret_val
            results.append(ret_val)
        # Merge the results together if they are string
        sample_res = results[0]
        if isinstance(sample_res, str):
            results = ' '.join(results)
        return results


def process_normal_query(query, data, inst_list, full_json, doc_extract):
    try:
        # Temporary workaround for placeholder, removing '?' from query
        logging.debug('Input query: %s' % query)
        san_query = query.replace('\n', '')
        # san_query = query.replace('?', '')
        # logging.debug('Sanitized input query: %s' % san_query)
        res = process_cus_input(get_input_dict(san_query, data), inst_list, doc_extract)
        if (not full_json) and ('extracted_triples' in res):
            # Step 2: Convert the `extracted_triples` string to a Python list
            extracted_triples = ast.literal_eval(res["extracted_triples"])

            # Step 3: Process each triple to make it valid JSON
            processed_triples = []
            for triple in extracted_triples:
                triple_dict = ast.literal_eval(triple.replace("\\'", "'"))
                if isinstance(triple_dict.get("triple"), set):  # Check if "triple" is a set
                    triple_dict["triple"] = tuple(triple_dict["triple"])  # Convert set to tuple
                processed_triples.append(triple_dict)

            # Step 4: Update the dictionary with the processed triples
            res["extracted_triples"] = processed_triples
            return json.dumps(res['extracted_triples'])
        return res
    except Exception as inst:
        logging.exception('Exception occurred for the query: %s\nException: %s' % (query, inst))
        return {}



def get_input_dict(san_query, data):
    rep_before = False
    if 'replace_before' in data:
        rep_before = eval(data['replace_before'])
    placeholder = def_placeholder
    if 'placeholder' in data:
        placeholder = data['placeholder']
    f_input = {
        'text': san_query,
        'replace_before': rep_before,
        'placeholder': placeholder
    }
    # Passing on all the params that were not yet modified and are not in the exclusion list
    for entry in data:
        if (entry not in f_input) and (entry not in io_exc_list):
            f_input[entry] = data[entry]

    return f_input
# Process custom pipeline requests
def process_cus_input(input_query, inst_list, doc_extract):
    logging.debug('Pipeline Info:\n%s' % inst_list)
    # Persist the input/output for the pipeline components
    io_var = input_query
    # Check the input language
    # check_lang(io_var)
    args = {
        '--text_tag': text_tag,
        '--dataset_path': input_file_path,
        '--entities_dict': entities_dict_path,
        '--error_ent_dict': error_ent_dict_path,
        '--relations_dict': relations_dict_path,
        '--output_IRIs_file': output_IRIs_file,
        '--dataset_type': dataset_type,
        '--text_tag_id': text_tag_id,
        '--short_claims': doc_extract
    }
    # Loop through pipeline components and pass it the previous output as an input
    for inst in inst_list:
        # Log start time
        start_time = time.time()
        if doc_extract:
            inst.process_document_api(io_var,args, entities_dict,error_ent_dict, relations_dict)
        else:
            inst.process_single_claim_api(io_var,args, entities_dict,error_ent_dict, relations_dict)
        # Print step time
        logging.debug('Time needed to process input using %s class: %s second(s)' % (type(inst).__name__, (time.time() - start_time)))
    # return the last output
    logging.info('final output: %s' % io_var)
    logging.info("\n\n")
    return io_var


def check_lang(input):
    query = input['text']
    if 'lang' not in input:
        lang = c_util.detect_lang(query)
        input['lang'] = lang


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)  # Specify port if different than default