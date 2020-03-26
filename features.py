import functions
import json


def calculate_features(input_signal):

    with open("./info.json") as handle:
        info_json = json.load(handle)    
    
    result = []

    for ft in info_json['features']['using']:
        aux = eval(info_json['features']['functions'][str(ft)])
        result.append(aux)
    
    return result