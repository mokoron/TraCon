from models.GTS import train as gts_train
from models.Graph_WaveNet import train as gwn_train
from models.HA import train as ha_train
from models.VAR import train as var_train
from models.MA import train as ma_train

import time


def run(config):
    t1 = time.time()
    trainers = {
        "GTS": gts_train,
        "Graph_WaveNet": gwn_train,
        "gwn": gwn_train,
        "HA": ha_train,
        "VAR": var_train,
        "ST_Norm": gwn_train,    # constraint: s_norm = 1 and t_norm = 1
        "MA": ma_train
    }
    try:
        metrics, bestid = trainers[config.model_name].run(config)
    except KeyError as e:
        raise AssertionError(
            f"Model {config.model_name} either is not correctly written or it is not in the list of handled models ")

    print(f"Training with {config.model_name} completed")

    t2 = time.time()
    time_spent = t2-t1
    print("Total time spent: {:.4f}".format(time_spent))
    return metrics, bestid, time_spent