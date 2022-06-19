import time

import pandas as pd
import models.util as models_util

def run(config):
    # objective: compute val and test predictions for required time horizons and save them in correct path.

    t1 = time.time()
    # load h5 data
    df = models_util.load_h5_data(config)

    # transform into df with id | date_time | speed
    columns = ["date_time", "id", "speed"]
    types_ = [{'id': 'int64'}]
    df = models_util.transform_data(df, columns, types_)

    # derive columns weekday and time
    df["weekday"] = df["date_time"].dt.dayofweek
    df["time"] = df["date_time"].dt.time

    # divide dataframe in train, val and test in the same proportions defined in config
    buffer = config.seq_length + config.output_horizon - 1
    train_val_test_df, train_val_test_num, _ = models_util.split_data(df=df, config=config, buffer=buffer)

    # in train df, compute column ha_speed
    train_df = pd.DataFrame({'ha_speed': train_val_test_df[0].groupby(["id", "weekday", "time"])["speed"].mean()}).reset_index()

    # in val and tet, compute ha_speed based on train (join)
    val_df = train_val_test_df[1].join(train_df.set_index(['id', 'weekday', 'time']), on=['id', 'weekday', 'time'])
    test_df = train_val_test_df[2].join(train_df.set_index(['id', 'weekday', 'time']), on=['id', 'weekday', 'time'])

    # for val and test, construct the historical average prediction data of shape (num_time_steps, output_horizon, num_nodes, 1)
    ## real values for metrics
    real_val = models_util.get_pivot_table(df=val_df, values="speed")
    real_test = models_util.get_pivot_table(df=test_df, values="speed")

    val_df = models_util.get_pivot_table(df=val_df)
    test_df = models_util.get_pivot_table(df=test_df)


    # for val and test, reshape the prediction in the correct format (num_samples/time_steps, num_sensor/nodes, horizon)
    preds_val = models_util.get_future_values(config, val_df, train_val_test_num[1])
    preds_test = models_util.get_future_values(config, test_df, train_val_test_num[2])
    ## real values for metrics
    real_val = models_util.get_future_values(config, real_val, train_val_test_num[1])
    real_test = models_util.get_future_values(config, real_test, train_val_test_num[2])

    t2 = time.time()

    # compute the metrics using gwn function (save val and test in respective location)
    metrics = models_util.metrics_final_print(preds_val, real_val, preds_test, real_test, config, (t2-t1))
    return metrics


