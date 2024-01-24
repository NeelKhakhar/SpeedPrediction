import numpy as np
import matplotlib.pyplot as plt
import os
import configparser
import pandas as pd


from sklearn.metrics import mean_squared_error
def normalize_A1(x_):
    x_[:,::,4] = x_[:,::,4]/100
    x_[:,::,1] = x_[:,::,1]/600
    x_[:,::,0] = x_[:,::,0]/100

    return x_



def normalize_A3(x_):
    x_[:,::,4] = x_[:,::,4]/100
    x_[:,::,1]= x_[:,::,1]/600
    x_[:,::,0] = x_[:,::,0]/100
    x_[:,::,5] = x_[:,::,5]/100
    x_[:,::,6] = (x_[:,::,6]+1000)/(7000+1000)
    x_[:,::,7] = (x_[:,::,7]+700)/(7300+700)
    x_[:,::,8] = x_[:,::,8]/100

    return x_

def normalize_A2(x_):
    x_[:,::,4] = x_[:,::,4]/100
    x_[:,::,1] = x_[:,::,1]/600
    x_[:,::,0] = x_[:,::,0]/100
    x_[:,::,5] = x_[:,::,5]/100
    x_[:,::,6] = x_[:,::,6]/100

    return x_

def other_predict(model, datalogger, ext):

    driver_res={}
    for driver in datalogger.config[8]:
        x_test, y_test = datalogger.other_test_split_sequences(driver)
        x_test = normalize_A3(x_test)
        configParser = configparser.RawConfigParser()
        configParser.read(r'config.txt')
        out_dir = os.path.join(configParser.get('config', 'driver_testing_data_path'), 'outputs\\' + driver+'\\')

        y_actual = np.array(y_test)
        y_predicted = np.array(model.predict(x_test, verbose=0)) *  100
        rmse = mean_squared_error(y_actual, y_predicted, squared=False)
        driver_res[driver] = [rmse]
        print(rmse)
        if not os.path.exists(out_dir[:-1]):
            os.makedirs(out_dir[:-1])
        np.savetxt(out_dir+ext+"_actual.txt", y_actual, delimiter=",")
        np.savetxt(out_dir+ext+"_prediction.txt", y_predicted, delimiter=",")

        # plot the prediction vs ground truth on test data
        plt.plot(y_actual, label='actual speed',color="r")
        plt.plot(y_predicted, label='predicted speed',color="g")
        plt.title("{}_window={}_horizon={}_rmse={}".format(driver, datalogger.config[0], datalogger.config[1], rmse))
        plt.xlim([-200,14500])
        plt.ylim([-2,70])
        plt.legend()
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.savefig(out_dir + ext + ".png")
        # plt.show()
        plt.close()
        print(len(y_predicted))
        print("{}_window={}_horizon={}_rmse={}".format(driver, datalogger.config[0], datalogger.config[1], rmse))
    df = pd.DataFrame.from_dict(driver_res)    
    df.to_csv(out_dir+ext+'.csv')