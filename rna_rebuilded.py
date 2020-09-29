import time
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wandb
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from wandb.keras import WandbCallback

from preprocessing import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class HyperParameter:
    def __init__(self, arguments):
        if arguments is None:
            self.default = True
            self.dropout = 0.2
            self.epochs = 1000
            self.optimizer = 'SGD'
            self.initializer = 'he_normal'
            self.layer_size_hl1 = 40
            self.layer_size_hl2 = 40
            self.layer_size_hl3 = 40
            self.activation = 'sigmoid'
            self.batch_size = 100
        else:
            self.default = False
            self.dropout = round(float(arguments.dropout), 2)
            self.epochs = int(arguments.epochs)
            self.optimizer = arguments.optimizer
            self.activation = arguments.activation
            self.initializer = arguments.initializer
            self.layer_size_hl1 = int(arguments.layer_size_hl1)
            self.layer_size_hl2 = int(arguments.layer_size_hl2)
            self.layer_size_hl3 = int(arguments.layer_size_hl3)
            self.batch_size = 100

    def get_dict(self):
        return dict(dropout=round(float(self.dropout), 2),
                    epochs=int(self.epochs),
                    optimizer=self.optimizer,
                    activation=self.activation,
                    initializer=self.initializer,
                    layer_size_hl1=int(self.layer_size_hl1),
                    layer_size_hl2=int(self.layer_size_hl2),
                    layer_size_hl3=int(self.layer_size_hl3)
                    )


def create_model(cfg: HyperParameter) -> Sequential:  # Return sequential model
    # Here is where the magic really happens! Check this out:
    model = Sequential()  # The model used is the sequential
    # It has a fully connected input layer
    model.add(Dense(number_of_features, activation=cfg.activation,
                    input_shape=(number_of_features,)))
    # With three others hidden layers
    model.add(Dense(cfg.layer_size_hl1, activation=cfg.activation))
    # And a dropout layer between them to avoid overfitting
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl2, activation=cfg.activation))
    model.add(Dropout(cfg.dropout))
    model.add(Dense(cfg.layer_size_hl3, activation=cfg.activation))
    model.add(Dense(len(modulation_list), activation='softmax'))
    model.compile(optimizer=cfg.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_rna(cfg: HyperParameter):
    X_train, X_test, y_train, y_test = preprocess_data()
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    new_id = str(uuid.uuid1()).split('-')[0]  # Generates a unique id to each RNA created

    X_train_tensor = tf.convert_to_tensor(
        X_train, dtype=np.float32, name='X_train'
    )
    X_test_tensor = tf.convert_to_tensor(
        X_test, dtype=np.float32, name='X_test'
    )
    # Create encoded labels using tensorflow's one-hot function (to use categorical_crossentropy)
    depth = len(modulation_list)
    y_train_tensor = tf.one_hot(y_train, depth)
    y_test_tensor = tf.one_hot(y_test, depth)

    model = create_model(cfg)
    # Once created, the model is then compiled, trained and saved for further evaluation
    # batch_size = number of samples per gradient update
    # verbose = 0 = silent, 1 = progress bar, 2 = one line per epoch
    if cfg.default:
        # Default mode
        history = model.fit(X_train_tensor, y_train_tensor, batch_size=cfg.batch_size, epochs=cfg.epochs, verbose=2)
    else:
        # Execute code only if wandb is activated
        history = model.fit(X_train_tensor, y_train_tensor, batch_size=1600, epochs=cfg.epochs, verbose=2,
                            callbacks=[WandbCallback(validation_data=(X_test_tensor, y_test_tensor))])

    model.save(str(join(rna_folder, 'rna-' + new_id + '.h5')))
    model.save_weights(str(join(rna_folder, 'weights-' + new_id + '.h5')))
    print(join("\nRNA saved with id ", new_id, "\n").replace("\\", ""))

    # Here is where we make the first evaluation of the RNA
    loss, acc = model.evaluate(X_test_tensor, y_test_tensor, batch_size=cfg.batch_size, verbose=2)
    print('Test Accuracy: %.3f' % acc)

    # Execute code only if wandb is activated
    if cfg.default is False:
        # Here, WANDB takes place and logs all metrics to the cloud
        metrics = {'accuracy': acc,
                   'loss': loss,
                   'dropout': cfg.dropout,
                   'epochs': cfg.epochs,
                   'initializer': cfg.initializer,
                   'layer_syze_hl1': cfg.layer_size_hl1,
                   'layer_syze_hl2': cfg.layer_size_hl2,
                   'layer_syze_hl3': cfg.layer_size_hl3,
                   'optimizer': cfg.optimizer,
                   'activation': cfg.activation,
                   'id': new_id}
        wandb.log(metrics)

    # Here we make a prediction using the test data...
    print('\nStarting prediction')
    predict = np.argmax(model.predict(X_test_tensor, batch_size=cfg.batch_size), axis=-1)

    # And create a Confusion Matrix for a better visualization!
    print('\nConfusion Matrix:')
    confusion_matrix = tf.math.confusion_matrix(tf.argmax(y_test_tensor, axis=1), predict).numpy()
    confusion_matrix_normalized = np.around(
        confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis],
        decimals=2)
    print(confusion_matrix_normalized)
    cm_data_frame = pd.DataFrame(confusion_matrix_normalized, index=modulation_list, columns=modulation_list)
    figure = plt.figure(figsize=(8, 4), dpi=150)
    sns.heatmap(cm_data_frame, annot=True, cmap=plt.cm.get_cmap('Blues', 6))
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(fig_folder, 'confusion_matrix-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_accuracy-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.clf()
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.savefig(join(fig_folder, 'history_loss-' + new_id + '.png'), bbox_inches='tight', dpi=300)

    plt.close(figure)


def evaluate_rna(input_id=' ', test_size=500):  # Make a prediction using some samples to evaluate the RNA behavior
    rna_folder = pathlib.Path(join(os.getcwd(), 'rna'))
    fig_folder = pathlib.Path(join(os.getcwd(), "figures"))
    data_folder = pathlib.Path(join(os.getcwd(), "mat-data", "pickle"))

    print("\nStarting RNA evaluation by SNR.")

    if input_id == ' ':  # If you do not specify a RNA id, it'll use the newest available in rna_folder
        aux = [f for f in os.listdir(rna_folder) if "rna" in f]
        rna_files = [join(str(rna_folder), item) for item in aux]
        input_id = max(rna_files, key=os.path.getctime)
        print("\nRNA ID not provided. Using RNA model with id {}, created at {} instead.".format(
            input_id.split("-")[1].split(".")[0], time.ctime(os.path.getmtime(input_id))))
        input_id = input_id.split("-")[1].split(".")[0]
        rna = join(str(rna_folder), "rna-" + input_id + ".h5")
        model = load_model(rna)  # Loads the RNA model
    else:
        rna = join(str(rna_folder), "rna-" + input_id + ".h5")
        model = load_model(rna)
        print("\nUsing RNA with id {}.".format(input_id))

    # For each modulation, randomly loads the test_size samples
    # and predict the result to all SNR values
    result = np.zeros((len(modulation_list), number_of_snr))
    for i, mod in enumerate(features_files):
        print("Evaluating {}".format(mod.split("_")[0]))
        with open(join(data_folder, mod), 'rb') as evaluating_data:
            data = pickle.load(evaluating_data)
        for snr in snr_list:
            random_samples = np.random.choice(data[snr - (21 - number_of_snr)][:].shape[0], test_size)
            X_test = [data[snr - (21 - number_of_snr)][i] for i in random_samples]
            # Instantiate StandardScaler
            scaler = StandardScaler()
            # Fit into data used for training, results are means and variances used to standardise the data
            X_test = scaler.fit_transform(X_test)
            right_label = [info_json['modulations']['labels'][mod.split("_")[0]] for _ in range(len(X_test))]
            predict = np.argmax(model.predict(X_test), axis=-1)
            accuracy = accuracy_score(right_label, predict)
            result[i][snr - (21 - number_of_snr)] = accuracy

    accuracy_graphic(result, fig_folder, input_id)

    if not os.path.isfile(join(rna_folder, "weights-" + input_id + ".h5")):
        print("Weights file not found. Saving it into RNA folder")
        model.save_weights(str(join(rna_folder, 'weights-' + input_id + '.h5')))


def accuracy_graphic(result, fig_folder, model_id):
    # Then, it creates an accuracy graphic, containing the
    # prediction result to all snr values and all modulations
    figure = plt.figure(figsize=(8, 4), dpi=150)
    plt.title("Accuracy")
    plt.ylabel("Right prediction")
    plt.xlabel("SNR [dB]")
    plt.xticks(np.arange(number_of_snr), [info_json['snr']['values'][i] for i in info_json['snr']['using']])
    for item in range(len(result)):
        plt.plot(result[item], label=modulation_list[item])
    plt.legend(loc='best')
    plt.savefig(join(fig_folder, "accuracy-" + model_id + ".png"),
                bbox_inches='tight', dpi=300)
    figure.clf()
    plt.close(figure)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='RNA argument parser')
    # parser.add_argument('--dropout', action='store', dest='dropout')
    # parser.add_argument('--epochs', action='store', dest='epochs')
    # parser.add_argument('--optimizer', action='store', dest='optimizer')
    # parser.add_argument('--initializer', action='store', dest='initializer')
    # parser.add_argument('--layer_size_hl1', action='store', dest='layer_size_hl1')
    # parser.add_argument('--layer_size_hl2', action='store', dest='layer_size_hl2')
    # parser.add_argument('--layer_size_hl3', action='store', dest='layer_size_hl3')
    # parser.add_argument('--activation', action='store', dest='activation')
    # arguments = parser.parse_args()
    # wandb.init(project="amcpy-team", config=HyperParameter(arguments).get_dict())
    # config = wandb.config
    train_rna(HyperParameter(None))
    evaluate_rna(' ')
