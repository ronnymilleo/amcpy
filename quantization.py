import pathlib
from os.path import join

import numpy as np
import scipy.io
import tensorflow as tf

from globals import arm_folder, q_range


def get_dense_layers(model: tf.keras.Model) -> tf.keras.layers:
    """
    Reads all layers from model and return only Dense layers
    """
    dense_layers = []
    for lyr in model.layers:
        if "dense" in lyr.name:
            dense_layers.append(lyr)
    return dense_layers


def find_best_q_format(min_value, max_value):
    if min_value > q_range['Q0.15'][0] and max_value < q_range['Q0.15'][1]:
        q_format = 'Q0.15'
    elif min_value > q_range['Q1.14'][0] and max_value < q_range['Q1.14'][1]:
        q_format = 'Q1.14'
    elif min_value > q_range['Q2.13'][0] and max_value < q_range['Q2.13'][1]:
        q_format = 'Q2.13'
    elif min_value > q_range['Q3.12'][0] and max_value < q_range['Q3.12'][1]:
        q_format = 'Q3.12'
    elif min_value > q_range['Q4.11'][0] and max_value < q_range['Q4.11'][1]:
        q_format = 'Q4.11'
    elif min_value > q_range['Q5.10'][0] and max_value < q_range['Q5.10'][1]:
        q_format = 'Q5.10'
    elif min_value > q_range['Q6.9'][0] and max_value < q_range['Q6.9'][1]:
        q_format = 'Q6.9'
    else:
        q_format = 'Input values require too many integer bits'
    return q_format


def quantize_data(input_array, q_format: str, q_type=tf.dtypes.qint16):
    f_min, f_max = q_range[q_format]
    quantized_data = tf.quantization.quantize(
        input_array, f_min, f_max, q_type, mode='SCALED'
    )
    return quantized_data


def quantize_rna(input_array, q_format_w: str, q_format_b: str, q_type=tf.dtypes.qint16):
    f_min_w, f_max_w = q_range[q_format_w]
    f_min_b, f_max_b = q_range[q_format_b]
    quantized_weights = tf.quantization.quantize(
        input_array[0], f_min_w, f_max_w, q_type, mode='SCALED',
        round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
        ensure_minimum_range=0.01
    )
    quantized_bias = tf.quantization.quantize(
        input_array[1], f_min_b, f_max_b, q_type, mode='SCALED',
        round_mode='HALF_AWAY_FROM_ZERO', name=None, narrow_range=False, axis=None,
        ensure_minimum_range=0.01
    )
    return list([quantized_weights, quantized_bias])


def dequantize_rna(input_array, q_format_w: str, q_format_b: str):
    f_min_w, f_max_w = q_range[q_format_w]
    f_min_b, f_max_b = q_range[q_format_b]
    dequantized_weights = tf.quantization.dequantize(
        input_array[0].output, f_min_w, f_max_w, mode='SCALED', name=None, axis=None,
        narrow_range=False, dtype=tf.dtypes.float32
    )
    dequantized_bias = tf.quantization.dequantize(
        input_array[1].output, f_min_b, f_max_b, mode='SCALED', name=None, axis=None,
        narrow_range=False, dtype=tf.dtypes.float32
    )
    return list([dequantized_weights, dequantized_bias])


def get_quantization_error(original_weights, dequantized_weights):
    err = original_weights - dequantized_weights
    return err


def quantize(model: tf.keras.Model, inputs):
    # Get weights and biases max and min floats before quantization
    d_layers = get_dense_layers(model)
    max_w = []
    max_b = []
    min_w = []
    min_b = []
    for dense_layer in d_layers:
        max_w.append(np.max(dense_layer.get_weights()[0]))
        max_b.append(np.max(dense_layer.get_weights()[1]))
        min_w.append(np.min(dense_layer.get_weights()[0]))
        min_b.append(np.min(dense_layer.get_weights()[1]))

    # Look for best Q number to represent each layer
    layer_dict = {}
    for n in range(len(max_w)):
        key = "Layer {} weights".format(n)
        layer_dict[key] = find_best_q_format(min_w[n], max_w[n])
        key = "Layer {} biases".format(n)
        layer_dict[key] = find_best_q_format(min_b[n], max_b[n])

    # Look for outputs of every layer
    layer_outputs = []
    layer_input = inputs
    for dense_layer in d_layers:
        layer_output = dense_layer(layer_input)
        layer_outputs.append(layer_output)
        layer_input = layer_output

    # Get max and min for each layer output
    max_outs = []
    min_outs = []
    for tensor in layer_outputs:
        max_outs.append(np.max(tensor[0]))
        min_outs.append(np.min(tensor[0]))

    # Figure out the precise Q number format for each output
    for n in range(len(max_outs)):
        key = "Layer {} outputs".format(n)
        layer_dict[key] = find_best_q_format(min_outs[n], max_outs[n])

    quantized = []
    dequantized_w = []
    a = 0
    for n, dense_layer in enumerate(d_layers):
        key = "Layer {} weights".format(n)
        q_w = layer_dict[key]
        key = "Layer {} biases".format(n)
        q_b = layer_dict[key]
        quantized.append(quantize_rna(dense_layer.get_weights(), q_w, q_b))
        dequantized_w.append(dequantize_rna(quantized[a], q_w, q_b))
        error_w = get_quantization_error(dense_layer.get_weights()[0], dequantized_w[a][0])
        print('Max error INPUT LAYER {} W: {}'.format(n, np.max(error_w)))
        error_b = get_quantization_error(dense_layer.get_weights()[1], dequantized_w[a][1])
        print('Max error INPUT LAYER {} B: {}'.format(n, np.max(error_b)))
        a = a + 1

    # TODO: better flattening of matrices
    # Convert quantized weights into numpy arrays
    l1 = np.reshape(quantized[0][0][0].numpy().T, (7 * 7,))
    b1 = quantized[0][1][0].numpy()
    l2 = np.reshape(quantized[1][0][0].numpy().T, (7 * 21,))
    b2 = quantized[1][1][0].numpy()
    l3 = np.reshape(quantized[2][0][0].numpy().T, (21 * 27,))
    b3 = quantized[2][1][0].numpy()
    l4 = np.reshape(quantized[3][0][0].numpy().T, (27 * 13,))
    b4 = quantized[3][1][0].numpy()
    l5 = np.reshape(quantized[4][0][0].numpy().T, (13 * 6,))
    b5 = quantized[4][1][0].numpy()

    weights = np.concatenate((l1, l2, l3, l4, l5))
    biases = np.concatenate((b1, b2, b3, b4, b5))
    save_dict = {'weights': weights, 'biases': biases}
    scipy.io.savemat(pathlib.Path(join(arm_folder, 'w_and_b.mat')), save_dict)
    return save_dict, layer_dict
