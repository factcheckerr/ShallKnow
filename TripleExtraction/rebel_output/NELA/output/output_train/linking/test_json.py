import json
import re

def add_backslash_to_quotes(input_string):
    # input_string = input_string.replace("'", "\'")
    print(input_string)
    input_string = re.sub(r'\\+', r'\\', input_string)
    input_string = re.sub(r'^[\'"]|[\'"]$', '', input_string)
    print(input_string)
    if input_string[0] == "'" and input_string[-1] == "'":
        input_string = input_string[1:-1]
    if input_string[0] == "'" and input_string[-1] != "'":
        input_string = input_string[1:]
    if input_string[0] != "'" and input_string[-1] == "'":
        input_string = input_string[:-1]

    if input_string[:2] == "\"'" and input_string[-2:] == "'\"":
        print(input_string)
        input_string = input_string[2:-2]
        print(input_string)

    if input_string[0] == "\"" and input_string[-1] == "\"":
        input_string = input_string[1:-1]
        print(input_string)

    input_string = input_string.replace("\xad", "")
    input_string = input_string.replace("\xa0", "")
    input_string = repr(input_string)
    # input_string = input_string.replace("\\\'", "\'")
    input_string = input_string.replace("\\u200e", "\u200e")

    input_string  = input_string.replace("\"","").replace("\\","")
    if input_string[0] != "\"" and input_string[-1] != "\"":
        # here the string is really without any quotations
        input_string = input_string.replace('"', '\\"')
        input_string = input_string.replace('\"', '\\"')
        input_string = "\"" + input_string + "\""


    if input_string[0] == "\"" and input_string[1] == "\"":
        input_string = input_string.replace("\"\"", "\"")
        print(input_string)
        print("!!!!!!!!check why here!!!!!")
        exit(1)

    input_string = input_string.replace("\\\\\"", "\"")
    input_string = input_string.replace(":", "\:").replace("\\","")
    print(input_string)
    return input_string

def read_jsonl_file(file_path):
    data2 = []
    with open(file_path, 'r', encoding='utf-8') as file,open('2-'+file_path, 'w', encoding='utf-8') as file2:
        for line in file:
            try:
                print(line)
                # print(add_backslash_to_quotes(line.split("\"id\":")[1][:-2]))
                # line=("{\"triple\": " + ((line.split("\"triple\":")[1].split("\"subject\":")[0][:-2])) +","
                # + " \"subject\": " + ((line.split("\"subject\":")[1].split("\"predicate\":")[0][:-2])) +","
                # + " \"predicate\": " + ((line.split("\"predicate\":")[1].split("\"object\":")[0][:-2])) +","
                #  + " \"object\": " + ((line.split("\"object\":")[1].split("\"claim\":")[0][:-2])) +","
                # + " \"claim\": " + add_backslash_to_quotes(line.split("\"claim\":")[1].split("\"id\":")[0][2:-4]) +","
                # + " \"id\":" + add_backslash_to_quotes(line.split("\"id\":")[1][:-2]) +"}")
                data = json.loads(line)
                data2.append(data)
                print(line)
                # yield data['id']
                # break
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(line)
                break
        for ll in data2:
            file2.write(str(ll)+"\n")
            # file2.write(str(ll))

# Example usage:
file_path = '2-output_triples_IRIs.jsonl'
read_jsonl_file(file_path)
# for json_data in read_jsonl_file(file_path):
#     print(json_data)