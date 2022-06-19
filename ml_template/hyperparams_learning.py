import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

metr_la = "METR-LA"
pems_bay = "PEMS-BAY"
hann_data = "HANNOVER-DATA"
datasets = [hann_data, metr_la, pems_bay]


def label_data(working_df, threshold, l, r_l):
    working_df = working_df.astype({'id': 'int64'}, {'date_time': 'datetime'})
    required_columns = ["id", "date_time", "time", "speed", "is_traffic_jam", "is_transition", "none"]

    # adding of a new concept: transition points (computation of speed-1 and speed-2)
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

    # continue with segment and time
    working_df["time"] = working_df["date_time"].dt.time

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
        (working_df['is_traffic_jam'] == 1) | (working_df['is_rush_hour'] == 1) | (working_df['is_transition'] == 1), 0,
        1)

    working_df["concept"] = np.where((working_df['is_transition'] == 1), 2,
                                     np.where((working_df['is_traffic_jam'] == 1), 1, 0))

    return working_df[required_columns]


if __name__ == "__main__":
    start_hour = 6
    end_hour = 22

    fold_number = 5
    selection = {}

    th_list = [0.6, 0.7, 0.8, 0.9]
    l_list = [0.1, 0.2, 0.3]
    r_l_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # r/l

    # compute labeled data for each combination
    for d in datasets:
        input_file = f'data/{d.lower()}.h5'

        df = pd.read_hdf(input_file)
        if df.isnull().values.any():
            df = df.fillna(80)
            print('Warning: NaN Values in Data found. Filled them with 80!')
        df = df.reset_index()
        index = df.columns[0]
        df = df[(df[index].dt.hour >= start_hour) & (df[index].dt.hour < end_hour)]
        df = pd.melt(df, id_vars=df.columns[0], value_vars=df.columns)
        df = df.rename(
            {df.columns[0]: 'date_time', df.columns[1]: 'id', df.columns[2]: 'speed'},
            axis='columns')

        combinations = {}
        for threshold in th_list:
            for l in l_list:
                for r_l in r_l_list:
                    name = f"thresh{threshold}_l{l}_rl{r_l}"
                    working_df = label_data(df, threshold, l, r_l)
                    combinations[name] = {"labeled_data": working_df}

        selection[d] = {"combinations": combinations}

    # compute f_score for each combination
    for d in datasets:
        Classifier = DecisionTreeClassifier

        classifier = Classifier()
        f_scores = []
        input_variables = ["id", "speed", "is_wd", "time_min"]
        target_variable = "concept"
        for threshold in th_list:
            for l in l_list:
                for r_l in r_l_list:
                    name = f"thresh{threshold}_l{l}_rl{r_l}"
                    df = selection[d]["combinations"][name]["labeled_data"]
                    pivot_df = pd.pivot_table(df, values='speed', index='date_time', columns=['id'], aggfunc=np.mean)
                    num_samples = pivot_df.shape[0]

                    folds = []
                    num_val = int(num_samples / fold_number)
                    num_train = num_samples - num_val
                    for f in range(fold_number):
                        val_df = pivot_df[f * num_val:(f + 1) * num_val]
                        train_df = pivot_df[:f * num_val][(f + 1) * num_val:]

                        train_datetimes, val_datetimes = train_df.index.tolist(), val_df.index.tolist()

                        train_data = df.loc[df["date_time"].isin(train_datetimes)][input_variables].to_numpy()
                        train_labels = df.loc[df["date_time"].isin(train_datetimes)][target_variable].to_numpy()

                        val_data = df.loc[df["date_time"].isin(val_datetimes)][input_variables].to_numpy()
                        real = df.loc[df["date_time"].isin(val_datetimes)][target_variable].to_numpy()

                        classifier.fit(train_data, train_labels)
                        pred = classifier.predict(val_data)

                        fold = {
                            "min": np.min(f1_score(pred, real, average=None)),
                            "scores": f1_score(pred, real, average=None),
                            "score": np.min(f1_score(pred, real,
                                                     average='macro'))  # = np.mean(f1_score(pred, real, average=None))
                        }

                        folds.append(fold)

                    m_min = np.mean([fold["min"] for fold in folds])
                    m_scores = np.mean([fold["scores"] for fold in folds], axis=1)
                    m_score = np.mean([fold["score"] for fold in folds])
                    f_score = {
                        "dataset": d,
                        "threshold": threshold,
                        "l": l,
                        "r_l": r_l,
                        "score": m_score,
                        "scores": m_scores,
                        "min": m_min
                    }

                    f_scores.append(f_score)

        selection[d]["f_scores"] = f_scores

    print(selection)
