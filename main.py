import calculate_features
import graphics
from preprocessing import preprocess_data


# Some lines of code might be commented, it's because I don't need to run then anymore
# If you get any errors, try running those lines
if __name__ == '__main__':
    # calculate_features.run()
    # graphics.plot()
    x_train, x_test, y_train, y_test, scaler = preprocess_data('Training')
