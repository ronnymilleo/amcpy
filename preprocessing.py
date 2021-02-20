import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from globals import *


def preprocess_data(dataset='Training'):
    if dataset == 'Training':
        # Allocate memory for all signals
        number_of_samples = number_of_training_frames * len(training_SNR)
        X = np.zeros((number_of_samples * len(signals), len(used_features)), dtype=np.float32)
        y = np.ndarray((number_of_samples * len(signals),), dtype=np.int8)
    else:
        # Allocate memory for all signals
        number_of_samples = number_of_testing_frames * len(testing_SNR)
        X = np.zeros((number_of_samples * len(signals), len(used_features)), dtype=np.float32)
        y = np.ndarray((number_of_samples * len(signals),), dtype=np.int8)

    # Here each modulation file is loaded and all frames to all SNR values are vertically stacked
    for i, mod in enumerate(features_files):
        print("Processing {} data".format(mod.split("_")[0]))  # Separate the word 'features' from modulation file
        data_dict = scipy.io.loadmat(join(data_folder, mod))
        data = data_dict[mat_info[mod.split("_")[0]]]
        # Location of each modulation on input matrix based on their number of samples
        location = i * number_of_samples

        for SNR in training_SNR:
            for frame in range(number_of_training_frames):
                X[location, :] = np.float32(data[SNR][frame][:])  # [SNR][frames][ft]
                location += 1

            # An array containing the encoded labels for each modulation
            start = i * number_of_samples
            end = start + number_of_samples
            for index in range(start, end):
                y[index] = i

    # Finally, the data is split into train and test samples and standardised for a better learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("\nData shape:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit into data used for training, results are means and variances used to standardise the data
    scaler.fit(X_train)
    # Remove mean and variance from data_train
    standardised_data_train = scaler.transform(X_train)
    # Remove mean and variance from data_test using the same values (based on theoretical background)
    standardised_data_test = scaler.transform(X_test)
    return standardised_data_train, standardised_data_test, y_train, y_test, scaler


def configure_data():
    # Allocate memory for all signals
    number_of_samples = number_of_testing_frames * len(testing_SNR)
    X = np.zeros((number_of_samples * len(signals), len(used_features)), dtype=np.float32)

    # Here each modulation file is loaded and all frames to all SNR values are vertically stacked
    for i, mod in enumerate(features_files):
        print("Processing {} data".format(mod.split("_")[0]))  # Separate the word 'features' from modulation file
        data_dict = scipy.io.loadmat(join(data_folder, mod))
        data = data_dict[mat_info[mod.split("_")[0]]]
        # Location of each modulation on input matrix based on their number of samples
        location = i * number_of_samples

        for SNR in testing_SNR:
            for frame in range(number_of_testing_frames):
                X[location, :] = np.float32(data[SNR][frame][:])  # [SNR][frames][ft]
                location += 1

    return X
