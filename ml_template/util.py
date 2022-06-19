import os
import numpy as np
import pandas as pd
import shutil

metr_la = "METR-LA"
pems_bay = "PEMS-BAY"
hann_data = "HANNOVER-DATA"


def label_data(working_df, threshold, rh, l, r_l):
    working_df = working_df.astype({'id': 'int64'})
    required_columns =["id", "date_time", "speed", "is_traffic_jam", "is_transition", "delta_speed"]

    test_df = pd.DataFrame(columns=working_df.columns)
    for segment in working_df.id.unique():
        df = working_df.loc[working_df['id'] == segment]
        df['speed-1'] = df['speed'].shift(1)
        df['speed+1'] = df['speed'].shift(-1)
        df['speed-2'] = df['speed'].shift(2)
        df['delta_speed'] = df['speed'] - df['speed-1']
        test_df = pd.concat([test_df, df], sort=True)
    test_df = test_df.fillna(-1)
    test_df = test_df.sort_index()
    working_df = test_df

    # time
    working_df["time"] = working_df["date_time"].dt.time
    # dow, Monday = 0, Sunday = 6
    working_df["dow"] = working_df["date_time"].dt.dayofweek

    # weekday (WD) VS week-end (WE)
    working_df["dow_type"] = np.where((working_df["dow"] == 5) | (working_df["dow"] == 6), "WE", "WD")

    # start with segment, taking into consideration dow type
    segment_df = pd.DataFrame({'median_speed_segment': working_df.groupby(["id", "dow_type"])["speed"].median()}).reset_index()

    # continue with segment and time
    segment_time_df = pd.DataFrame(
        {'median_speed_segment_time': working_df.groupby(["id", "time", "dow_type"])["speed"].median()}).reset_index()

    # add median_speed_segment and mmedian_speed_segment_time in working_df
    working_df = working_df.join(segment_df.set_index(['id', 'dow_type']), on=['id', 'dow_type'])
    working_df = working_df.join(segment_time_df.set_index(['id', 'time', 'dow_type']), on=['id', 'time', 'dow_type'])

    # definition of concepts: No overlapping
    working_df["is_transition"] = np.where((working_df['speed-1'] < 0), 0, np.where(
        (np.abs(working_df['speed'] - working_df['speed-1']) > ((l) * working_df['median_speed_segment'])) & (
                np.abs(working_df['speed'] - working_df['speed-2']) > (
                (l) * working_df['median_speed_segment'])) & (
                np.abs(working_df['speed-1'] - working_df['speed+1']) > r_l * (
                (l) * working_df['median_speed_segment'])), 1, 0))

    working_df['is_traffic_jam'] = np.where(working_df['is_transition'] == 1, 0, np.where(
        ((working_df['speed-1'] < 0) & (
                    (1 / 2) * (working_df['speed'] + working_df['speed+1']) < threshold * working_df[
                        "median_speed_segment"])), 1,
        np.where(((working_df['speed+1'] < 0) & (
                    (1 / 2) * (working_df['speed-1'] + working_df['speed']) < threshold * working_df[
                        "median_speed_segment"])),
                 1, np.where(((working_df['speed-1'] > 0) & (working_df['speed+1'] > 0) & (
                    (1 / 3) * (working_df['speed-1'] + working_df['speed'] + working_df['speed+1']) < threshold *
                    working_df["median_speed_segment"])),
                             1, 0))))

    working_df["none"] = np.where(
        (working_df['is_traffic_jam'] == 1) |
        (working_df['is_transition'] == 1), 0,
        1)

    return working_df[required_columns]


def get_categories(df, threshold, rh, l, r_l):
    working_df = df
    working_df = pd.melt(working_df, id_vars=working_df.columns[0], value_vars=df.columns)
    working_df = working_df.rename(
        {working_df.columns[0]: 'date_time', working_df.columns[1]: 'id', working_df.columns[2]: 'speed'},
        axis='columns')
    working_df = label_data(working_df, threshold, rh, l, r_l)
    jam_mx = pd.pivot_table(working_df, values='is_traffic_jam', index='date_time', columns=['id'], aggfunc=np.mean)
    transition_mx = pd.pivot_table(working_df, values='is_transition', index='date_time', columns=['id'],
                                   aggfunc=np.mean)
    none_mx = pd.pivot_table(working_df, values='none', index='date_time', columns=['id'], aggfunc=np.mean)
    train_mx = pd.pivot_table(working_df, values='speed', index='date_time', columns=['id'], aggfunc=np.mean)
    categories_mx = pd.merge(jam_mx, transition_mx, 'left', on='date_time')  # jam  | transition
    categories_mx = pd.merge(none_mx, categories_mx, 'left', on='date_time')  # none | jam | transition

    return train_mx, categories_mx


def get_root():
    root = 'ml_template/'
    while not os.path.exists(f"{root}"):
        root = f'../{root}'
    return root


def path_input_data(name=metr_la):
    input_data = 'ml_template/data'
    return f'{input_data}/{name.lower()}.h5'


def path_training_data(name=metr_la):
    training_data = 'ml_template/training_data'
    return f'{training_data}/{name.upper()}'


def path_adj_data(name='METR-LA'):
    adj_data = 'ml_template/sensor_graph'
    return f'{adj_data}/{name.upper()}'


def path_save_iter(name='METR-LA'):
    save_iter = 'ml_template/save'
    return f'{save_iter}/{name.upper()}'


def delete_not_matching(path, name):
    if not os.path.exists(path):
        # probably we are not saving the results
        return
    for folder_name in os.listdir(path):
        file_path = os.path.join(path, folder_name)
        try:
            if folder_name != name:
                if os.path.isfile(file_path):  # in case it is a file, not a folder
                    os.remove(file_path)
                elif os.path.islink(file_path):  # in case it is a link
                    os.unlink(file_path)
                elif os.path.isdir(file_path):  # in case it is a folder
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

