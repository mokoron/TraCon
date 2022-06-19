from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import pandas as pd
import util as u
import pickle
from models import util as models_util


def generate_graph_seq2seq_io_data_concept(
        df, categories, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, num_concepts=4
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape 
    data = np.reshape(df.values, (num_samples, 1, -1)).transpose((0,2,1)) # data.shape = (num_samples, num_nodes, 1)     1 --> speed
    categories = np.reshape(categories.values, (num_samples, num_concepts, -1)).transpose((1,0,2)) # data.shape = (4, num_samples, num_nodes)  4 for none, jam, rush, transition
    data_list = [data]
    df.index = pd.to_datetime(df.index)
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)) # time_in_day.shape = (num_samples, num_nodes, 1)
        data_list.append(time_in_day)
      
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)
   
    data = np.concatenate(data_list, axis=-1)   # data.shape = (num_samples, num_nodes, num_values+1)
    x, y, c = [], [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        c_t = categories[:, t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
        c.append(c_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    c = np.stack(c, axis=0)
    print("x shape: ", x.shape, ", y shape: ", y.shape, ", c shape: ", c.shape, ", data shape: ", data.shape)
    return x, y, c  # x/y.shape = (num_samples, seq_length, num_nodes, num_values+1)
                    # c.shape = (num_samples, num_concepts, seq_length, num_nodes)
                    # concepts: none | jam | transition


def generate_train_val_test_concept(config, output_dir):
    seq_length_x, seq_length_y = config.seq_length, config.output_horizon
    df = models_util.load_h5_data(config)

    # set parameters after learning
    threshold = {
        u.hann_data: 0.8,
        u.metr_la: 0.6,
        u.pems_bay: 0.6
    }
    rh = {
        u.hann_data: 0.95,
        u.metr_la: 0.9,
        u.pems_bay: 0.9
    }
    l = {
        u.hann_data: 0.2,
        u.metr_la: 0.2,
        u.pems_bay: 0.2
    }
    r_l = {
        u.hann_data: 0.5,
        u.metr_la: 0.5,
        u.pems_bay: 0.5
    }
    dataset = config.dataset_name.upper()
    df, categories = u.get_categories(df, threshold[dataset], rh[dataset], l[dataset], r_l[dataset])

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(config.y_start, (seq_length_y + 1), 1))

    x, y, c = generate_graph_seq2seq_io_data_concept(
        df,
        categories,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
        num_concepts=int(categories.shape[1]/df.shape[1])
    )

    num_samples = x.shape[0]
    num_test = int(num_samples * config.test_fraction)
    num_train = int(num_samples * config.train_fraction)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train, c_train = x[:num_train], y[:num_train], c[:num_train]
    # val
    x_val, y_val, c_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
        c[num_train: num_train + num_val],
    )
    # test
    x_test, y_test, c_test = x[-num_test:], y[-num_test:], c[-num_test:]
    
    for cat in ["train", "val", "test"]:
        _x, _y, _c = locals()["x_" + cat], locals()["y_" + cat], locals()["c_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "c:", _c.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            c=_c, 
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    
    return df
