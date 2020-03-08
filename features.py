import functions
import json


def calculate_features(input_signal):

    with open("./info.json") as handle:
        infoJson = json.load(handle)    
    
    result = []

    for ft in infoJson['features']['using']:
        aux = eval(infoJson['features']['functions'][str(ft)])
        result.append(aux)
    
    return result