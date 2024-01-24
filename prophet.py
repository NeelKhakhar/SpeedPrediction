import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import math


def prophet_predict_fb(observed_data, x_name="ds", y_name="y", forecast_cnt=365, frep="D", file_name=""):
    """
    function that predict time series with library fbprophet
    :param observed_data: time series data(DataFrame format)
    (two columns, one is time in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format and the other is numeric data)
    :param x_name: x column name(time data), usually is DATE
    :param y_name: y column name(numeric data) e.g. HMD, MAX...
    :param forecast_cnt: how many point needed to be predicted
    :param frep: the frequency/period of prediction
    :param file_name:
    :return: None
    """

    def check_parameter_validity():
        if x_name not in observed_data.keys():
            raise KeyError("train_data doesn't have column named %s" % x_name)
        if y_name not in observed_data.keys():
            raise KeyError("train_data doesn't have column named %s" % y_name)

    try:
        check_parameter_validity()
    except KeyError as e:
        print("key error: %s" % str(e))
        return None

    observed_data = observed_data.rename(columns={x_name: "ds", y_name: "y"})

    observed_data["ds"] = pd.to_datetime(observed_data["ds"])
    observed_data["y"] = pd.to_numeric(observed_data["y"], downcast='float', errors='coerce')

    df2_pro = fbprophet.Prophet(changepoint_prior_scale=0.1)
    df2_pro.fit(observed_data)

    future_date = df2_pro.make_future_dataframe(periods=forecast_cnt, freq=frep)
    df2_forecast = df2_pro.predict(future_date)

    # register a datetime converter for matplotlib
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    if file_name:
        fig1 = df2_pro.plot(df2_forecast, xlabel=x_name, ylabel=y_name)
        fig1.show()
        fig1.savefig('./result/%s.png' % file_name)
        fig2 = df2_pro.plot_components(df2_forecast)
        fig2.show()
        fig2.savefig('./result/%s.png' % str(file_name + "1"))

    return df2_forecast

