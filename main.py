import argparse

import numpy as np
import wandb

import quantization
from neural_network import NNConfig, get_model_from_id, confusion_matrix, evaluate_nn
from preprocessing import preprocess_data
from serial_comm import serial_communication

# Some lines of code might be commented, it's because I don't need to run then anymore
# If you get any errors, try running those lines
if __name__ == '__main__':
    # Are you training or testing?
    training = False
    # In case you're using a microcontroller
    use_microcontroller = False

    # calculate_features.run()
    # graphics.plot()

    # Weights and biases configuration
    if training:
        x_train, x_test, y_train, y_test, scaler = preprocess_data('Training')

        # Weights and biases parser
        parser = argparse.ArgumentParser(description='ANN Argument Parser')
        parser.add_argument('--activation', action='store', dest='activation')
        parser.add_argument('--batch_size', action='store', dest='batch_size')
        parser.add_argument('--dropout', action='store', dest='dropout')
        parser.add_argument('--epochs', action='store', dest='epochs')
        parser.add_argument('--layer_size_hl1', action='store', dest='layer_size_hl1')
        parser.add_argument('--layer_size_hl2', action='store', dest='layer_size_hl2')
        parser.add_argument('--layer_size_hl3', action='store', dest='layer_size_hl3')
        parser.add_argument('--learning_rate', action='store', dest='learning_rate')
        parser.add_argument('--optimizer', action='store', dest='optimizer')
        args = parser.parse_args()

        neuralnet = NNConfig(None)
        wandb.init(project="amcpy-team", config=neuralnet.get_dict())
        config = wandb.config
        model_id = neuralnet.train(x_train, x_test, y_train, y_test)
        loaded_model, loaded_model_id = get_model_from_id(model_id)
        evaluate_nn(loaded_model, loaded_model_id, scaler)
        confusion_matrix(loaded_model, loaded_model_id, x_test, y_test)

    if not training:
        x_train, x_test, y_train, y_test, scaler = preprocess_data('Test')
        loaded_model, loaded_model_id = get_model_from_id('model-best')
        evaluate_nn(loaded_model, loaded_model_id, scaler)
        confusion_matrix(loaded_model, loaded_model_id, x_test, y_test)

    if use_microcontroller:
        x_train, x_test, y_train, y_test, scaler = preprocess_data('Test')
        loaded_model, loaded_model_id = get_model_from_id('model-best')
        load_dict, info_dict = quantization.quantize(loaded_model, np.concatenate((x_train, x_test)))
        for info in info_dict:
            print(info + ' -> ' + info_dict[info])
        weights = load_dict['weights']
        biases = load_dict['biases']
        serial_communication(weights, biases, scaler, info_dict)
