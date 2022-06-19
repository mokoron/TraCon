import time
import numpy as np
import torch
import models.util as models_util
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR


def run(config):
    # objective: compute val and test predictions for required time horizons and save them in correct path.

    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')
    t1 = time.time()
    # load h5 data
    df = models_util.load_h5_data(config)
    index_column = df.columns[0]
    buffer = config.seq_length
    _, [num_train, num_val, num_test], df = models_util.split_data(df=df, config=config, index_column=index_column, buffer=buffer)
    df = df.set_index(index_column)

    # define parameter d for each time series and corresponding derivative
    d, df_diff = get_stationary(df)  # df_diff.shape = (num_samples, num_nodes)

    # compute prediction for all time horizons in shape (num_samples, num_nodes, horizon)
    preds_val_diff, preds_test_diff, constant_cols = get_predictions(config, df_diff,
                                                                     df, num_train, num_val, num_test)  # shape = (num_samples, horizon, num_nodes)
    preds_val_diff, preds_test_diff = preds_val_diff.transpose([0, 2, 1]), preds_test_diff.transpose([0, 2, 1])


    # invert and get result in torch tensor
    preds_val, preds_test = invert_prediction(d, preds_val_diff, preds_test_diff, device=device)

    # for val and test, compute real format (num_samples / time_steps, num_sensor / nodes, horizon)
    real = torch.tensor(df.values).to(device)
    real_val = models_util.get_future_values(config, real[num_train - buffer: num_train + num_val], num_val)
    real_test = models_util.get_future_values(config, real[-num_test - buffer:], num_test)

    t2 = time.time()

    # compute the metrics using gwn function (save val and test in respective location).
    metrics = models_util.metrics_final_print(preds_val, real_val, preds_test, real_test, config, (t2 - t1))
    return metrics


def get_stationary(df):
    num_nodes = df.shape[1]
    # num_nodes = 2
    # define parameter d for each time series and corresponding derivative
    d = np.zeros(num_nodes)

    df_diff = []
    for i in range(num_nodes):
        diff = df.iloc[:, i]
        df_diff.append(diff)
    df_diff = np.stack(df_diff, axis=0).transpose([1, 0])

    return d, df_diff


def get_predictions(config, df_diff, df, num_train, num_val, num_test):
    # split df_diff in train_diff, val_diff and test_diff in the same proportions as in config, adding buffer for val and test
    buffer = config.seq_length

    train_diff = df_diff[:num_train]  # shape = (num_samples, num_nodes)
    val_diff = df_diff[num_train - buffer: num_train + num_val]
    test_diff = df_diff[-num_test - buffer:]

    # train the model (using train_diff)
    min_p = 1
    max_p = 10
    p = min_p
    best_aic = None
    best_model = None
    cols = []
    for i in range(min_p, max_p + 1):
        # df_no_constants = train_diff.loc[:, (train_diff != train_diff.iloc[0]).any()]
        model = VAR(train_diff)
        try:
            results = model.fit(i)
        except Exception as e:
            cols = extract_constant_cols(e.args[0])
            train_deleted = np.stack(train_diff[:, cols], axis=0)
            val_deleted = np.stack(val_diff[:, cols], axis=0)
            test_deleted = np.stack(test_diff[:, cols], axis=0)
            train_diff = np.delete(train_diff, cols, 1)
            val_diff = np.delete(val_diff, cols, 1)
            test_diff = np.delete(test_diff, cols, 1)

            model = VAR(train_diff)
            results = model.fit(i)

        aic = results.aic
        if i == min_p:
            best_aic = aic
            best_model = results
        elif aic < best_aic:
            p = 1
            best_aic = aic
            best_model = results

    # compute pred_diff for val and test for all time steps
    val_outputs = []
    test_outputs = []

    # val
    for t in range(num_val):
        # compute pred_diff for val passing output_horizon, shape = (horizon, num_nodes), for only one time step
        val_output = best_model.forecast(y=val_diff[t:t + buffer], steps=config.output_horizon)
        val_outputs.append(val_output)

    # test
    for t in range(num_test):
        # compute pred_diff for test passing output_horizon, shape = (horizon, num_nodes), for only one time step
        test_output = best_model.forecast(y=test_diff[t:t + buffer], steps=config.output_horizon)
        test_outputs.append(test_output)

    preds_val_diff = np.stack(val_outputs, axis=0)
    preds_test_diff = np.stack(test_outputs, axis=0)  # shape = (num_samples, horizon, nodes)

    # insert removed columns if there are
    if len(cols) > 0:
        preds_val_diff = insert_columns(preds_val_diff, val_deleted, cols, config.output_horizon)
        preds_test_diff = insert_columns(preds_test_diff, test_deleted, cols, config.output_horizon)

    return preds_val_diff, preds_test_diff, cols


def invert_prediction(d, preds_val_diff, preds_test_diff,
                      device):  # preds_test_diff.shape = (num_samples, num_nodes, horizon)
    preds_val = []
    preds_test = []
    for i in range(len(d)):
        der = d[i]
        val = preds_val_diff[:, i, :]
        test = preds_test_diff[:, i, :]
        while der > 0:
            # too complex to implement, maybe if we have more time

            der -= 1

    preds_val = torch.tensor(preds_val_diff).to(device)
    preds_test = torch.tensor(preds_test_diff).to(device)

    return preds_val, preds_test


def extract_constant_cols(message):
    # message = "x contains one or more constant columns. Column(s) 23, 125, 127 are constant. Adding a constant with trend='c' is not allowed."
    first_marker = "Column(s)"
    last_marker = "are constant."

    # get end first marker
    begin = message.index(first_marker) + len(first_marker)

    # get beginning last marker
    end = message.index(last_marker)

    # extract substring
    sub = message[begin:end]
    values = sub.split(",")

    # convert into integer array
    cols = [int(values[i]) for i in range(len(values))]

    return cols


def insert_columns(input_array, columns, cols, horizons):
    # isert the columns at the correct position in input array for all horizons
    # input_array.shape = (num_samples, horizons, nodes-num_cols)
    # columns.shape = (num_samples+buffer, len(cols))
    # output_array.shape = (num_samples, horizons, num_nodes)

    columns = columns[-input_array.shape[0]:]
    output_array = input_array
    for i in range(len(cols)):
        values = np.stack([np.stack([columns[:, i] for _ in range(horizons)], axis=1)], axis=-1)     # shape = (num_samples, horizons, 1)
        output_array = np.concatenate([output_array[:, :, :cols[i]], values, output_array[:, :, cols[i]:]], axis=-1)

    return output_array