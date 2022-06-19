import time

import torch
import numpy as np
import models.util as models_util


def run(config):
    # objective: compute val and test predictions for required time horizons and save them in correct path.
    """
    to predict time t* to (t+n)* with window k, we predict t* using t-1 to t-k,
    then we predict (t+1)* using t*, t-1 to t-k+1; and so on.
    for values at time t to t+output_horizon-1 we only need t-1 to t-window; for real we need t to t+output_horizon-1
    """

    t1 = time.time()
    # load h5 data
    df = models_util.load_h5_data(config)

    # transform into df with id | date_time | speed
    columns = ["date_time", "id", "speed"]
    types_ = [{'id': 'int64'}]
    df = models_util.transform_data(df, columns, types_)

    # divide dataframe in train, val and test in the same proportions defined in config
    buffer = config.seq_length + config.output_horizon - 1
    index_column = columns[0]
    train_val_test_df, train_val_test_num, _ = models_util.split_data(df, config, index_column, buffer)

    # for val and test, construct the initial tensor, with previous (seq_length) speeds of each (u, t)
    ## real values for metrics
    real_val = models_util.get_pivot_table(df=train_val_test_df[1], values="speed")
    real_test = models_util.get_pivot_table(df=train_val_test_df[2], values="speed")

    initial_val = get_initial_tensor(real_val, config)
    initial_test = get_initial_tensor(real_test, config)

    # for val and test, construct predictions of each (u, t) of shape (num_samples, num_nodes, horizon)
    preds_val = get_moving_average(initial_val, config.seq_length, config.output_horizon, train_val_test_num[1])
    preds_test = get_moving_average(initial_test, config.seq_length, config.output_horizon, train_val_test_num[2])

    # for val and test, compute real format (num_samples / time_steps, num_sensor / nodes, horizon)
    real_val = models_util.get_future_values(config, real_val, train_val_test_num[1])
    real_test = models_util.get_future_values(config, real_test, train_val_test_num[2])

    t2 = time.time()

    # compute the metrics using gwn function (save val and test in respective location).
    metrics = models_util.metrics_final_print(preds_val, real_val, preds_test, real_test, config, (t2-t1))
    return metrics


def get_initial_tensor(df, config, index_column='date_time'):
    x_offsets = np.sort(np.concatenate((np.arange(-(config.seq_length - 1), 1, 1),)))
    # y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))
    min_t = abs(min(x_offsets))
    num_samples = df.shape[0]
    values = models_util.get_values(df, num_samples, config.output_horizon, min_t, x_offsets)

    return values


def get_moving_average(initial, window, output_horizon, num_samples):
    final = torch.tensor([])
    for i in range(output_horizon):
        intermediate = torch.cat([initial, final], dim=1)
        new = torch.mean(intermediate[:, -window:, :, :], 1)
        new = new.unsqueeze(dim=1)
        final = torch.cat([final, new], dim=1)
    preds = final.squeeze().permute(0, 2, 1)

    # remove buffered samples for val and test
    preds = preds[-num_samples:, ...]    # shape = (num_samples, num_nodes, horizon)

    return preds

