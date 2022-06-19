import torch
from models.Graph_WaveNet import util as gwn_util
import os
import numpy as np
import pandas as pd
import util as u
import threading

lock = threading.Lock()


sum_types = ["linear", "quadratic", "cubic", "exp", "lin_exp"]


def save_element(element, params, elt_id, partition_id):
    if params[f"save_{elt_id}"]:
        path_dir = f'{elt_id}/{params["phase"]}/{params["model_name"]}/{params["dataset_name"]}/{params["epoch"]}{partition_id}'
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(element, os.path.join(path_dir, f'horizon_{params["horizon"]}.pt'))


def load_element(params, elt_id, partition_id):
    path_dir = f'{elt_id}/{params["phase"]}/{params["model_name"]}/{params["dataset_name"]}/{params["epoch"]}{partition_id}'
    element = torch.load(os.path.join(path_dir, f'horizon_{params["horizon"]}.pt'))
    return element


def clean_directory(config, index, phase, rename=False, name=None):
    model_name = name if name is not None else config.model_name
    best_folder = f"epoch_{index}"
    path_to_loss = f'loss/{phase}/{model_name}/{config.dataset_name}'
    path_to_pred_real = f'pred_real/{phase}/{model_name}/{config.dataset_name}'
    new_name = f"epoch_0"
    if (config.save_loss and model_name.lower() != "classification") or (model_name.lower() == "classification" and config.label_save_loss):
        u.delete_not_matching(path_to_loss, best_folder)
        if rename:
            os.rename(os.path.join(path_to_loss, best_folder), os.path.join(path_to_loss, new_name))
    if (config.save_pred_real and model_name.lower() != "classification") or (model_name.lower() == "classification" and config.label_save_pred_real):
        u.delete_not_matching(path_to_pred_real, best_folder)
        if rename:
            os.rename(os.path.join(path_to_pred_real, best_folder), os.path.join(path_to_pred_real, new_name))


def load_h5_data(config):
    with lock:
        root = u.get_root()
        input_file = f'{root}{config.input_dir}/{config.dataset_name.lower()}.h5'
        df = pd.read_hdf(input_file)
        if df.isnull().values.any():
            df = df.fillna(80)
            print('Warning: NaN Values in Data found. Filled them with 80!')

        df = df.reset_index()
        index = df.columns[0]
        df = df[(df[index].dt.hour >= config.start_hour) & (df[index].dt.hour < config.end_hour)]

        return df


def transform_data(df, columns, types_):
    df = pd.melt(df, id_vars=df.columns[0], value_vars=df.columns)
    df = df.rename(
        {df.columns[0]: columns[0], df.columns[1]: columns[1], df.columns[2]: columns[2]},
        axis='columns')
    for type_ in types_:
        df = df.astype(type_)
    return df


def split_data(df, config, index_column="date_time", buffer=0):
    # extraction of matching samples used to generate split data
    x_offsets = np.sort(np.concatenate((np.arange(-(config.seq_length - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(config.y_start, (config.output_horizon + 1), 1))

    num_samples = len(df[index_column].unique())

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

    matching_samples = df[index_column].unique()[min_t:max_t]

    df = df.loc[df[index_column].isin(matching_samples)]

    # now we can get the samples of each part
    num_samples = len(df[index_column].unique())

    num_test = int(num_samples * config.test_fraction)
    num_train = int(num_samples * config.train_fraction)
    num_val = num_samples - num_test - num_train

    # the train, val and test df for historical average computation
    training_datetimes = df[index_column].unique()[:num_train]
    train_df = df.loc[df[index_column].isin(training_datetimes)]
    val_datetimes = df[index_column].unique()[num_train - buffer: num_train + num_val]  # buffer added
    val_df = df.loc[df[index_column].isin(val_datetimes)]
    test_datetimes = df[index_column].unique()[-(num_test + buffer):]  # buffer added
    test_df = df.loc[df[index_column].isin(test_datetimes)]

    return [train_df, val_df, test_df], [num_train, num_val, num_test], df


def get_pivot_table(df, index_column="date_time", node_column="id", values='ha_speed'):
    df = pd.pivot_table(df, values=values, index=index_column, columns=node_column,
                        aggfunc=np.mean)
    num_samples = df.shape[0]
    df = np.reshape(df.values, (num_samples, 1, -1)).transpose(
        (0, 2, 1))  # data.shape = (num_samples, num_nodes, 1)     1 --> prediction

    return df


def get_values(df, samples, output_horizon, min_t, offsets):
    max_t = abs(samples - output_horizon)  # Exclusive
    values = []
    for t in range(min_t, max_t):
        values_t = df[t + offsets, ...]
        values.append(values_t)
    values = torch.tensor(np.stack(values, axis=0))

    return values


def metrics_final_print(preds_val, real_val, preds_test, real_test, config, time_spent):
    mval_amae, mval_amape, mval_armse = get_metrics(preds_val, real_val, config, "val")
    mtest_amae, mtest_amape, mtest_armse = get_metrics(preds_test, real_test, config, "test")

    # return metrics
    log = '{} - dataset: {}. Validation results: Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}'
    print(log.format(config.model_name, config.dataset_name, mval_amae, mval_amape, mval_armse, time_spent), flush=True)

    log = '{} - dataset: {}. Test results: Test Loss: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}'
    print(log.format(config.model_name, config.dataset_name, mtest_amae, mtest_amape, mtest_armse), flush=True)

    return [mtest_amae, mtest_amape, mtest_armse], 0


def get_metrics(preds, realy, config, phase):
    epoch = {"val": f"epoch_{0}/", "test": ""}
    amae = []
    amape = []
    armse = []
    output_horizon = np.min([config.output_horizon, config.seq_length])
    for i in range(output_horizon):
        pred = preds[:, :, i]
        real = realy[:, :, i]
        params = {"phase": phase, "model_name": config.model_name, "dataset_name": config.dataset_name, "horizon": i,
                  "epoch": epoch[phase], "save_loss": config.save_loss, "save_pred_real": config.save_pred_real}
        metrics = gwn_util.metric(pred, real, params=params)
        if phase == "test":
            log = '{} - dataset: {}. Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(config.model_name, config.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    return np.mean(amae), np.mean(amape), np.mean(armse)


def get_future_values(config, df, num_samples):
    # x_offsets = np.sort(np.concatenate((np.arange(-(config.seq_length - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(config.y_start, (config.output_horizon + 1), 1))
    min_t = 0
    samples = df.shape[0]

    values = get_values(df, samples, config.output_horizon, min_t, y_offsets)
    values = values.squeeze().permute(0, 2, 1)

    # remove buffered samples for val and test
    values = values[-num_samples:, ...]

    return values


def weighted_sum(weights, x_list, type_=sum_types[0]):
    weights = torch.tensor(weights)
    # num = f(i) is tensor --> unsqueeze(dim) --> get list of tensors --> concatenate along dim --> sum along dim.
    if type_ == sum_types[0]:   # linear
        denom = torch.sum(weights)
        num = torch.sum(torch.cat([(weights[i]*x_list[i]).unsqueeze(dim=0) for i in range(weights.shape[0])], dim=0), dim=0)
    elif type_ == sum_types[1]: # quadratic
        denom = torch.sum(torch.pow(weights, weights))
        num = torch.sum(torch.cat([((weights[i] ** 2) * x_list[i]).unsqueeze(dim=0) for i in range(weights.shape[0])], dim=0), dim=0)
    elif type_ == sum_types[2]:     # cubic
        denom = torch.sum(torch.pow(weights, 3))
        num = torch.sum(torch.cat([((weights[i] ** 3) * x_list[i]).unsqueeze(dim=0) for i in range(weights.shape[0])], dim=0), dim=0)
    elif type_ == sum_types[3]:     # exponential
        denom = torch.sum(torch.exp(weights))
        num = torch.sum(torch.cat([(torch.exp(weights[i]) * x_list[i]).unsqueeze(dim=0) for i in range(weights.shape[0])], dim=0), dim=0)
    elif type_ == sum_types[4]:     # xe**x
        denom = torch.sum(torch.pow(weights, torch.exp(weights)))
        num = torch.sum(torch.cat([((weights[i]*torch.exp(weights[i])) * x_list[i]).unsqueeze(dim=0) for i in range(weights.shape[0])], dim=0), dim=0)

    output = num / denom
    return output