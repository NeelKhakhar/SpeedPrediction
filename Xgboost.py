import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import math


def xgboost_predict(train_data, evaluation_data, forecast_cnt=365, freq="D", importance_fig=False, model_dir=""):
    """
    predict time series with XGBoost library which is based on Gradient Boost and CART(classification and regression tree)
    :param train_data: data use to train the model
    :param evaluation_data: data use to evaluate the model
    :param forecast_cnt: how many point needed to be predicted
    :param freq: the interval between time index
    :param importance_fig: whether plot importance of each feature
    :param model_dir: directory of pre-trained model(checkpoint, params)
    :return:
    """

    def create_features(df, label=None):
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X

    model_directory = "./model/XGBoost_%s" % now()
    params = {

    }
    # if there is a pre-trained model, use parameters from it
    if model_dir:
        model_directory = model_dir

    latest_date = evaluation_data["ds"].tolist()[-1]
    # set index with datetime
    train_data = train_data.set_index(pd.DatetimeIndex(train_data["ds"]))
    evaluation_data = evaluation_data.set_index(pd.DatetimeIndex(evaluation_data["ds"]))
    forecast_data = pd.DataFrame.from_dict({
        "ds": generate_time_series(start_date=latest_date, cnt=forecast_cnt, delta=delta_dict[freq])
    })
    forecast_data = forecast_data.set_index(pd.DatetimeIndex(forecast_data["ds"]))

    x_train, y_train = create_features(train_data, label='y')
    x_eval, y_eval = create_features(evaluation_data, label="y")
    x_forecast = create_features(forecast_data)

    reg = XGBRegressor(n_estimators=1000)
    if model_dir:
        reg.load_model(model_directory)
    else:
        reg.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_eval, y_eval)],
                early_stopping_rounds=50,
                verbose=False)  # Change verbose to True if you want to see it train
        reg.save_model(model_directory)

    if importance_fig:
        plot_importance(reg, height=0.9)

    evaluation_data["y"] = reg.predict(x_eval)
    forecast_data["y"] = reg.predict(x_forecast)

    return evaluation_data, forecast_data

