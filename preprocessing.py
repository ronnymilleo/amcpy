import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from globals import *


def preprocess_data(dataset: str):
    if dataset == 'Training':
        snr_axis = training_snr
    else:
        snr_axis = testing_snr

    number_of_samples = number_of_frames * len(snr_axis)
    X = np.zeros((number_of_samples * len(signals), len(used_features)), dtype=np.float32)
    y = np.ndarray((number_of_samples * len(signals),), dtype=np.int8)
    z = np.ndarray((number_of_samples * len(signals),), dtype=np.int8)

    index = 0
    for _ in signals:
        for snr in snr_axis:
            for frame in range(0, number_of_frames):
                z[index] = snr
                index = index + 1

    # Here each modulation file is loaded and all frames to all SNR values are vertically stacked
    for i, mod in enumerate(features_files):
        print("Processing {} data".format(mod.split("_")[0]))  # Separate the word 'features' from modulation file
        data_dict = scipy.io.loadmat(join(matlab_data_folder, mod))
        data = data_dict[mat_info[mod.split("_")[0]]]
        # Location of each modulation on input matrix based on their number of samples
        location = i * number_of_samples

        for snr in snr_axis:
            for frame in range(number_of_frames):
                X[location, :] = np.float32(data[snr][frame][:])  # [SNR][frames][ft]
                location += 1

            # An array containing the encoded labels for each modulation
            start = i * number_of_samples
            end = start + number_of_samples
            for index in range(start, end):
                y[index] = i

    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit into data used for training, results are means and variances used to standardise the data
    scaler.fit(X)
    # Remove mean and variance from the dataset
    transformed_X = scaler.transform(X)

    # Finally, the data is split into train and test samples and standardised for a better learning
    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2, random_state=42)
    print("\nData shape:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, scaler


def configure_data():
    # Allocate memory for all signals
    number_of_samples = number_of_frames * len(testing_snr)
    X = np.zeros((number_of_samples * len(signals), len(used_features)), dtype=np.float32)
    y = np.ndarray((number_of_samples * len(signals),), dtype=np.int8)
    SNR_position = 0

    for snr in testing_snr:
        for i, mod in enumerate(features_files):
            print("Processing {} data".format(mod.split("_")[0]))  # Separate the word 'features' from modulation file
            data_dict = scipy.io.loadmat(join(matlab_data_folder, mod))
            data = data_dict[mat_info[mod.split("_")[0]]]
            # Location of each modulation on input matrix based on their number of samples
            location = i * number_of_frames + SNR_position
            # print(location)
            for frame in range(number_of_frames):
                X[location, :] = np.float32(data[snr][frame][:])  # [SNR][frames][ft]
                location += 1

            # An array containing the encoded labels for each modulation
            start = i * number_of_frames + SNR_position
            end = start + number_of_frames
            for index in range(start, end):
                y[index] = i

        SNR_position = SNR_position + number_of_frames * len(signals)

    return X, y
