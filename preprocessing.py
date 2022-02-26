import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from globals import *


# This function is used to reshape the training data and standardize it
def preprocess_data(dataset: str):
    if dataset == 'Training':
        snr_axis = training_snr
    else:
        snr_axis = all_available_snr

    number_of_samples = number_of_frames * len(snr_axis) * len(modulation_signals_with_noise)
    x = np.zeros((number_of_samples, len(used_features)))  # x is the known variable used for input (features)
    y = np.zeros(number_of_samples, )  # y is the known variable used for output (labels)

    for modulation_number, modulation in enumerate(features_files):  # Modulation + noise
        print("Processing {} data".format(modulation.split("_")[0]))  # Split the word 'features' from modulation file
        data_dict = scipy.io.loadmat(join(calculated_features_folder, modulation))
        data = data_dict[mat_info[modulation.split("_")[0]]]
        # Location of each modulation on input matrix based on their number of samples
        index = modulation_number * number_of_frames * len(snr_axis)

        # Put all features, from all SNR and all frames in the same matrix
        for snr in snr_axis:
            for frame in range(0, number_of_frames):
                x[index, :] = np.float32(data[snr][frame][used_features])  # [SNR][frames][features] 32 bits
                index += 1

        # Map every group of 18 features to a modulation result
        start = modulation_number * number_of_frames * len(snr_axis)
        end = start + number_of_frames * len(snr_axis)
        for i in range(start, end):
            y[i] = modulation_number

    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit into data used for training, results are means and variances used to standardise the data
    scaler.fit(x)
    # Remove mean and variance from the dataset
    scaled_x = scaler.transform(x)

    # Finally, the data is split into train and test samples and standardised for a better learning
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=42)
    print("\nData shape:")
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    return x_train, x_test, y_train, y_test, scaler


# This function is used to reshape the calculated feature's data to be used in the neural network
def reshape_data():
    number_of_samples = number_of_frames * len(all_available_snr) * len(modulation_signals_with_noise)
    x = np.zeros((number_of_samples, len(used_features)))  # x is the known variable used for input (features)
    y = np.zeros((number_of_samples, len(used_features)))  # y is the known variable used for output (labels)

    for modulation_number, modulation in enumerate(features_files):  # Modulation + noise
        print("Processing {} data".format(modulation.split("_")[0]))  # Split the word 'features' from modulation file
        data_dict = scipy.io.loadmat(join(calculated_features_folder, modulation))
        data = data_dict[mat_info[modulation.split("_")[0]]]
        # Location of each modulation on input matrix based on their number of samples
        index = modulation_number * number_of_frames * len(all_available_snr)

        # Put all features, from all SNR and all frames in the same matrix
        for snr in all_available_snr:
            for frame in range(0, number_of_frames):
                x[index, :] = np.float32(data[snr][frame][:])  # [SNR][frames][features] 32 bits

        # Map every group of 18 features to a modulation result
        start = modulation_number * number_of_frames * len(all_available_snr)
        end = start + number_of_frames * len(all_available_snr)
        for i in range(start, end):
            y[i] = modulation_number

    return x, y
