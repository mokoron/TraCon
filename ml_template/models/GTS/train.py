from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import yaml
from models.GTS.model.pytorch.supervisor import GTSSupervisor
import util as u


def main(config, config_filename):
    root = u.get_root()
    filepath = f'{root}ml_template/models/GTS/data/model/{config_filename}'
    with open(filepath) as f:
        supervisor_config = yaml.safe_load(f)
        supervisor = GTSSupervisor(config=config, **supervisor_config)
        metrics, bestid = supervisor.train()
        return metrics, bestid


def run(config):
    if config.dataset_name.upper() == 'PEMS-BAY'.upper():
        config_filename = 'para_bay.yaml'
    elif config.dataset_name.upper() == 'METR-LA'.upper():
        config_filename = 'para_la.yaml'
    else:
        if config.dataset_name.upper() == 'braunschweig'.upper():
            print('braunschweig data not available for this model, we will use HANNOVER data')
        config_filename = 'para_hannover.yaml'

    metrics, bestid = main(config, config_filename)
    return metrics, bestid
