import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import math


def now():
    return datetime.now().strftime("%m_%d_%H_%M_%s")


def parse_result_tf(tf_data):
    """
    parse the result of model output in tensorflow
    :param tf_data: the output of tensorflow
    :return: data in DataFrame format
    """
    return pd.DataFrame.from_dict({"ds": tf_data["times"].reshape(-1), "y": tf_data["mean"].reshape(-1)})


def generate_time_series(
        start_date=datetime(2006, 1, 1),
        cnt=4018, delta=timedelta(days=1), timestamp=False
):
    """
    generate a time series/index
    :param start_date: start date
    :param cnt: date count. If =cnt are specified, delta must not be; one is required
    :param delta: time delta, default is one day.
    :param timestamp: output timestamp or format string
    :return: list of time string or timestamp
    """

    def per_delta():
        curr = start_date
        while curr < end_date:
            yield curr
            curr += delta

    end_date = start_date + delta * cnt

    time_series = []
    if timestamp:
        for t in per_delta():
            time_series.append(t.timestamp())
    else:
        for t in per_delta():
            time_series.append(t)
        # print(t.strftime("%Y-%m-%d"))
    return time_series


def AR_predict_tf(train_data, evaluation_data, forecast_cnt=365, freq="D", model_dir=""):
    """
    predict time series with auto-regressive model in tensorflow
    :param train_data: data use to train the model
    :param evaluation_data: data use to evaluate the model
    :param forecast_cnt: how many point needed to be predicted
    :param freq: the interval between time index
    :param model_dir: directory of pre-trained model(checkpoint, params)
    :return:
    """
    model_directory = "./model/AR_%s" % now()
    params = {
        # periodicities of the input data, in the same units as the time feature.
        # Note this can be a single value or a list of values for multiple periodicities.
        "periodicities": 52,
        # Number of past time steps of data to look at when doing the regression
        "input_window_size": 12,
        # Number of future time steps to predict. Note that setting it to > 1 empirically seems to give a better fit
        "output_window_size": 5,
        # The dimensionality of the time series (one for univariate, more than one for multivariate)
        "num_features": 1,
        # how many steps we train the model
        "global_steps": 3000
    }
    # if there is a pre-trained model, use parameters from it
    if model_dir:
        model_directory = model_dir
        params = read_model_param(model_dir + "/params.txt")

    # create time index for model training(use int)
    time_int = range(len(train_data) + len(evaluation_data))

    data_train = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: time_int[:len(train_data)],
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: train_data["y"],
    }

    data_eval = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: time_int[len(train_data):],
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: evaluation_data["y"],
    }

    reader_train = NumpyReader(data_train)
    reader_eval = NumpyReader(data_eval)

    """
    define in tensorflow/contrib/timeseries/python/timeseries/input_pipeline.py
    Note window_size must equal to input_window_size + output_window_size
    """
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader_train, batch_size=20, window_size=params["input_window_size"] + params["output_window_size"]
    )

    """
    define in tensorflow.contrib.timeseries.python.timeseries.estimators
    periodicities: periodicities of the input data, in the same units as the time feature. 
                   Note this can be a single value or a list of values for multiple periodicities
    num_features: The dimensionality of the time series (one for univariate, more than one for multivariate
    website: https://www.tensorflow.org/api_docs/python/tf/contrib/timeseries/ARRegressor
    """
    estimator_ar = tf.contrib.timeseries.ARRegressor(
        periodicities=params["periodicities"],
        input_window_size=params["input_window_size"],
        output_window_size=params["output_window_size"],
        num_features=params["num_features"],
        model_dir=model_directory,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS
    )

    # only train the model when there is no pre-trained model
    if not model_dir:
        estimator_ar.train(input_fn=train_input_fn, steps=params["global_steps"])

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader_eval)
    evaluation = estimator_ar.evaluate(input_fn=evaluation_input_fn, steps=1)
    # Predict starting after the evaluation
    (predictions,) = tuple(estimator_ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=forecast_cnt)))

    save_model_param(model_directory, params)
    if "loss" in evaluation.keys():
        # mean loss per mini-batch
        print("loss:%.5f" % evaluation["loss"])
        f = open(model_directory + "/%s" % evaluation["loss"], "w")
        f.close()
        model_log(
            evaluation["loss"],
            average_loss=-1 if "average_loss" not in evaluation.keys() else evaluation["average_loss"],
            content=model_dir
        )

    evaluation = parse_result_tf(evaluation)
    predictions = parse_result_tf(predictions)
    # here we should add an offset which is related to window size due to the inherent attribute of AR
    first_date = evaluation_data["ds"].tolist()[0] + \
                 delta_dict[freq] * (params["input_window_size"] + params["output_window_size"])
    evaluation["ds"] = generate_time_series(first_date, cnt=len(evaluation), delta=delta_dict[freq])
    latest_date = evaluation["ds"].tolist()[-1]
    predictions["ds"] = generate_time_series(latest_date, cnt=len(predictions), delta=delta_dict[freq])

    return evaluation, predictions
