import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import math

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.visualization import plot_data, box_plot, distribution_plot
from utils.visualization import heatmap, bar_plot, pie_plot, scatter_plot


def get_attributes(attributes_path):
    '''loads the attributes name'''
    with open(attributes_path) as f:
        attribute_names = f.readlines()
    attribute_names = [c.strip() for c in attribute_names]
    return attribute_names


class Preprocessing:
    _defaults = {
        "data_path"      : 'data/A_type0/type0_170.csv',
        "attribute"      : 'Transversal acceleration',
        "attribute_path" : 'attributes.txt',
        "sample_rate"    : 50,
        "norm_mtd"       : "MinMax",
        "norm_range"     : (-1, 1),
        "split_ratiio"   : (8, 1, 1),
        "moving_window"  : 25
    }


    def __init__(self, **kwargs): 
        self.__dict__.update(self._defaults)
        self.signal = self._get_signal()
        self.norm_signal = self.normalize()


    def _get_signal(self):
        assert self.sample_rate in [50, 10, 1], 'Sample rate must be 50, 10 or 1'
        assert self.attribute in get_attributes(self.attribute_path), 'Invalid attribute name'

        data_path = os.path.expanduser(self.data_path)
        assert data_path.endswith('.csv')

        data = pd.read_csv(data_path)
        attribute = self.attribute
        signal = data[attribute].dropna()
        return signal


    def normalize(self):
        """
        normalize the data
        :param method: support MinMax scaler or Z-Score scaler
        :param feature_range: use in MinMax scaler
        :return: normalized data(list), scaler
        """
        data = self.signal
        data = np.array(data)
        if len(data.shape) == 1 or data.shape[1] != 1:
            # reshape(-1, 1) --> reshape to a one column n rows 
            # matrix(-1 means not sure how many row)
            data = data.reshape(-1, 1)
        if self.norm_mtd == "MinMax":
            scaler = MinMaxScaler(feature_range=self.norm_range)
        elif self.norm_mtd == "Z-Score":
            scaler = StandardScaler()
        else:
            raise ValueError("only support MinMax scaler and Z-Score scaler")
        scaler.fit(data)
        # scaler transform apply to each column respectively
        # (which means that if we want to transform a 1-D data, we must 
        # reshape it to n x 1 matrix)
        return scaler.transform(data).reshape(-1)


    def denormalize(self):
        """
        denormalize data by scaler
        store this function here and later will move to post_processing/visulization
        :return: denormalized data
        """
        data = self.normalize()
        if self.norm_mtd == "MinMax":
            scaler = MinMaxScaler(feature_range=self.norm_range)
        elif self.norm_mtd == "Z-Score":
            scaler = StandardScaler()
        else:
            raise ValueError("only support MinMax scaler and Z-Score scaler")
        if len(data.shape) == 1 or data.shape[1] != 1:
            data = data.reshape(-1, 1)
        # max, min, mean, variance are all store in scaler, so we need it
        # to perform inverse transform
        return scaler.inverse_transform(data).reshape(-1)


    def resample(self, period="W"):
        """
        resample the original data to reduce noise
        :param period: the period of data e.g. B - business day, 
            D - calendar day, W - weekly, Y - yearly etc.
        (reference: pandas DateOffset Objects'http://pandas.pydata.org
            /pandas-docs/stable/user_guide/timeseries.html')
        :return:
        """
        data = data.set_index(pd.DatetimeIndex(data['ds']))
        return data.resample(period, label="right").mean().reset_index()


    def moving_avg(self):
        """
        smooth the data using the rolling mean method
        :param data:
        :param window: int, moving window size
        :return: smoothed data processed by the moving average
        """
        data = self.signal
        rolling = data.rolling(window=self.moving_window)
        rolling_mean = rolling.mean()
        # print(rolling_mean.head(10))
        # data.plot(color='blue')
        # rolling_mean.plot(color='red')
        # plt.show()
        return rolling_mean


    def close_session(self):
        self.sess.close()


if __name__ == "__main__":
    preprocessing = Preprocessing()
    # preprocessing.signal.plot()
    # preprocessing.moving_avg().plot(color='green')
    # plt.title('Smoothed Accelerator pedal operation of T170_1')
    # plt.show()
    norm = preprocessing.normalize()
    plot_data(norm)
    plot_data(preprocessing.signal)
