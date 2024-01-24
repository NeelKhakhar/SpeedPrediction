from keras.models import load_model
from data_logger import Datalogger
from model.multivariate.mlp import MIMLP_Attr01, MLP_Attr01
from model.multivariate.lstm import MILSTM
import numpy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import configparser


def model_input(X, y):
    X1 = X[:, :, 0]
    X2 = X[:, :, 1]
    X3 = X[:, :, 2]
    X4 = X[:, :, 3]
    X5 = X[:, :, 4]
    return [X1, X2, X3, X4, X5], y


if __name__ == "__main__":

    data_logger = Datalogger()
    window_size = data_logger.config[0]
    horizon = data_logger.config[1]
    configParser = configparser.RawConfigParser()
    configParser.read(r'config.txt')
    weight_pth = configParser.get('config', 'weights_file')
    out_dir = configParser.get('config', 'predict_save_pth')
    file_name = configParser.get('config', 'file_name')
    weight_file = weight_pth + file_name

    x_test, y_test = data_logger.test_split_sequences()
# FOR A1
    x_test[:,::,4] = x_test[:,::,4]/100
    x_test[:,::,1]= x_test[:,::,1]/600
    x_test[:,::,0] = x_test[:,::,0]/100
    len_test = len(x_test)

    backbone = "MILSTM"

    assert backbone in ["MLP", "MIMLP", "MILSTM"]
    
    if backbone == "MLP":
        x_test, y_test = model_input(x_test, y_test)
        # out_dir = "trained_weights_data/MLP_Attr01/"
    elif backbone == "MIMLP":
        x_test, y_test = model_input(x_test, y_test)
        # out_dir = "trained_weights_data/MTMLP_Attr01/"
    elif backbone == "MILSTM":
        # out_dir = "trained_weights_data/MILSTM_Attr01/"
        x_test = x_test
        y_test = y_test

    model = load_model(weight_file)
    print(model.summary())
    y_actual = numpy.array(y_test)
    y_predicted = numpy.array(model.predict(x_test, verbose=0))
    rmse = mean_squared_error(y_actual, y_predicted, squared=False)
    print(rmse)

    numpy.savetxt(out_dir+file_name[:-3]+"_actual.csv", y_actual, delimiter=",")
    numpy.savetxt(out_dir+file_name[:-3]+"_prediction.csv", y_predicted, delimiter=",")

    # plot the prediction vs ground truth on test data
    # plt.plot(y_actual, label='actual speed')
    # plt.plot(y_predicted, label='predicted speed')
    # plt.title("window={}_horizon={}_rmse={}".format(window_size, horizon, rmse))
    # plt.show()