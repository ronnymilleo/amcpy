import time
import uuid

import pandas as pd
import scipy.io
import seaborn as sns
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from wandb.keras import WandbCallback

from globals import *


class NNConfig:
    def __init__(self, arguments):
        if arguments is None:
            self.default = True
            self.activation = 'relu'
            self.batch_size = 32
            self.dropout = 0.4
            self.epochs = 10
            self.initializer = 'he_normal'
            self.layer_size_hl1 = 26
            self.layer_size_hl2 = 29
            self.layer_size_hl3 = 30
            self.learning_rate = 1e-3
            self.optimizer = 'rmsprop'
        else:
            self.default = False
            self.activation = arguments.activation
            self.batch_size = int(arguments.batch_size)
            self.dropout = round(float(arguments.dropout), 2)
            self.epochs = int(arguments.epochs)
            self.initializer = 'he_normal'
            self.layer_size_hl1 = int(arguments.layer_size_hl1)
            self.layer_size_hl2 = int(arguments.layer_size_hl2)
            self.layer_size_hl3 = int(arguments.layer_size_hl3)
            self.learning_rate = float(arguments.learning_rate)
            self.optimizer = arguments.optimizer

    def get_dict(self):
        return dict(activation=self.activation,
                    batch_size=int(self.batch_size),
                    dropout=round(float(self.dropout), 2),
                    epochs=int(self.epochs),
                    initializer=self.initializer,
                    layer_size_hl1=int(self.layer_size_hl1),
                    layer_size_hl2=int(self.layer_size_hl2),
                    layer_size_hl3=int(self.layer_size_hl3),
                    learning_rate=self.learning_rate,
                    optimizer=self.optimizer
                    )

    def train(self, x_train, x_test, y_train, y_test):
        new_id = str(uuid.uuid1()).split('-')[0]  # Generates a unique id to each RNA created
        # Create tensors from dataset
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=np.float32, name='X_train')
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=np.float32, name='X_test')
        # Create encoded labels using tensorflow's one-hot function (to use categorical_crossentropy)
        depth = len(modulation_signals_with_noise)
        y_train_tensor = tf.one_hot(y_train, depth)
        y_test_tensor = tf.one_hot(y_test, depth)
        # Instantiate model
        model = create_model(self)
        # Once created, the model is then compiled, trained and saved for further evaluation
        # batch_size = number of samples per gradient update
        # verbose >> 0 = silent, 1 = progress bar, 2 = one line per epoch

        # Execute code only if wandb is activated
        history = model.fit(x_train_tensor, y_train_tensor,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=2,
                            callbacks=[WandbCallback(validation_data=(x_test_tensor, y_test_tensor))],
                            validation_data=(x_test_tensor, y_test_tensor),
                            use_multiprocessing=True,
                            workers=8)

        model.save(str(join(trained_ann_folder, 'rna-' + new_id + '.h5')))
        model.save_weights(str(join(trained_ann_folder, 'weights-' + new_id + '.h5')))
        print(join("\nRNA saved with id ", new_id, "\n").replace("\\", ""))

        # Here is where we make the first evaluation of the RNA
        loss, acc = model.evaluate(x_train_tensor,
                                   y_train_tensor,
                                   batch_size=self.batch_size,
                                   verbose=2)
        print('Test Accuracy: {}'.format(acc))

        # Here, WANDB takes place and logs all metrics to the cloud
        metrics = {'accuracy': acc,
                   'loss': loss,
                   'activation': self.activation,
                   'batch_size': self.batch_size,
                   'dropout': self.dropout,
                   'epochs': self.epochs,
                   'initializer': self.initializer,
                   'layer_size_hl1': self.layer_size_hl1,
                   'layer_size_hl2': self.layer_size_hl2,
                   'layer_size_hl3': self.layer_size_hl3,
                   'learning_rate': self.learning_rate,
                   'optimizer': self.optimizer,
                   'id': new_id}
        wandb.log(metrics)

        # Here we make a prediction using the test data...
        print('\nStarting prediction')
        predict = np.argmax(model.predict(x_test), axis=-1)

        # And create a Confusion Matrix for a better visualization!
        print('\nConfusion Matrix:')
        cm = tf.math.confusion_matrix(y_test, predict).numpy()
        confusion_matrix_normalized = np.around(
            cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        print(confusion_matrix_normalized)

        cm_data_frame = pd.DataFrame(confusion_matrix_normalized,
                                     index=modulation_signals_with_noise,
                                     columns=modulation_signals_with_noise)

        figure = plt.figure(figsize=(8, 4), dpi=150)
        sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
        plt.tight_layout()
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(join(figures_folder, 'cm-' + new_id + '.png'), bbox_inches='tight', dpi=300)

        plt.clf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='best')
        plt.savefig(join(figures_folder, 'history_accuracy-' + new_id + '.png'), bbox_inches='tight', dpi=300)

        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='best')
        plt.savefig(join(figures_folder, 'history_loss-' + new_id + '.png'), bbox_inches='tight', dpi=300)

        plt.close(figure)
        return new_id


def evaluate_nn(evaluate_loaded_model, loaded_model_id, scaler):  # Make a prediction using some samples
    print("\nStarting RNA evaluation by snr.")

    # For each modulation, randomly loads the test_size samples
    # and predict the result to all snr values
    result = np.zeros((len(modulation_signals_with_noise), len(all_available_snr)))
    for i, mod in enumerate(features_files):
        print("Evaluating {}".format(mod.split("_")[0]))
        data_dict = scipy.io.loadmat(join(calculated_features_folder, mod))
        data = data_dict[mat_info[mod.split("_")[0]]]
        for snr in all_available_snr:
            x_dataset = data[snr, :, used_features]  # Test over all available data
            x_dataset = np.transpose(x_dataset)
            # Fit into data used for training, results are means and variances used to standardise the data
            x_dataset = scaler.transform(x_dataset)
            right_label = [modulation_signals_labels[i] for _ in range(len(x_dataset))]
            predict = np.argmax(evaluate_loaded_model.predict(x_dataset), axis=-1)
            accuracy = accuracy_score(right_label, predict)
            result[i][snr] = accuracy

    accuracy_graphic(result, loaded_model_id)
    save_dict = {'acc': result}
    scipy.io.savemat(pathlib.Path(join(os.getcwd(), 'figures', loaded_model_id + '_figure_data' + '.mat')), save_dict)

    if not os.path.isfile(join(trained_ann_folder, "weights-" + loaded_model_id + ".h5")):
        print("Weights file not found. Saving it into RNA folder")
        evaluate_loaded_model.save_weights(str(join(trained_ann_folder, 'weights-' + loaded_model_id + '.h5')))


def accuracy_graphic(result, loaded_model_id):
    # Then, it creates an accuracy graphic, containing the
    # prediction result to all snr values and all modulations
    plt.figure(figsize=(6, 3), dpi=300)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("SNR [dB]")
    plt.xticks(np.arange(len(all_available_snr)), [snr_values[i] for i in all_available_snr])
    for item in range(len(result)):
        plt.plot(result[item] * 100, label=modulation_signals_with_noise[item])
    plt.legend(loc='best')
    plt.savefig(join(figures_folder, "accuracy-" + loaded_model_id + ".png"),
                bbox_inches='tight', dpi=300)


def confusion_matrix(input_model, input_model_id, x_test, y_test):
    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = np.argmax(input_model.predict(x_test), axis=-1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    cm = tf.math.confusion_matrix(y_test, predict).numpy()
    confusion_matrix_normalized = np.around(
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(cm)
    print(confusion_matrix_normalized)

    cm_data_frame = pd.DataFrame(confusion_matrix_normalized,
                                 index=modulation_signals_with_noise,
                                 columns=modulation_signals_with_noise)

    plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(figures_folder, 'cm-' + input_model_id + '_' + str(len(all_available_snr)) + '.png'),
                bbox_inches='tight', dpi=300)


def get_model_from_id(model_str: str):
    if model_str == ' ':  # If you do not specify an RNA id, it'll use the newest available in trained_ann_folder
        aux = [f for f in os.listdir(trained_ann_folder) if "rna" in f]
        rna_files = [join(str(trained_ann_folder), item) for item in aux]
        input_id = max(rna_files, key=os.path.getctime)
        print("\nRNA ID not provided. Using RNA model with id {}, created at {} instead.".format(
            input_id.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(input_id))))
        input_id = input_id.split("-")[1].split(".")[0]
        neural_network = join(str(trained_ann_folder), "rna-" + input_id + ".h5")
        model = load_model(neural_network)  # Loads the RNA model
    else:
        neural_network = join(str(trained_ann_folder), "rna-" + model_str + ".h5")
        model = load_model(neural_network)
        print("\nUsing RNA with id {}.".format(model_str))
    return model, model_str


def create_model(cfg: NNConfig) -> Sequential:  # Return sequential model
    model = Sequential()
    model.add(Dense(len(used_features),
                    activation=cfg.activation,
                    input_shape=(len(used_features),),
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl1,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl2,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl3,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(len(modulation_signals_with_noise), activation='softmax'))

    if cfg.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == 'adam':
        optimizer = Adam(learning_rate=cfg.learning_rate)
    else:
        optimizer = Nadam(learning_rate=cfg.learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
