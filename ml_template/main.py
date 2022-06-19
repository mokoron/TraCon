import os
from configuration import Configuration, register_run
import train
import generate_training_data
from postgre_db import PostgreDB
from evaluation import Evaluation
import util as u
import threading
from models.Our_model import train as om_train


def setup_config(config):
    print('Configuration setup ...')

    config.add_entry("mN", "model_name", "model for the experiment")
    config.add_entry("dN", "dataset_name", "dataset for the experiment")
    config.add_entry("oDP", "only_data_preparation", "True if we want only to create train, val, test data", type=int)
    config.add_entry("sLs", "save_loss", "True if we want to save loss for val, test", type=int)
    config.add_entry("sPR", "save_pred_real", "True if we want to save pred_real for val, test", type=int)
    config.add_entry("tnF", "train_fraction", "fraction training samples", type=float)
    config.add_entry("ttF", "test_fraction", "fraction test samples", type=float)
    config.add_entry("ep", "epochs", "number of epochs", type=int)
    config.add_entry("th", "threshold", "traffic jam threshold speed", type=float)
    config.add_entry("l", "l", "left coefficient of transition", type=float)
    config.add_entry("rL", "r_l", "fraction given by r/l coefficient of transition", type=float)
    config.add_entry("lR", "learning_rate", "leaning rate of the model", type=float)
    config.add_entry("bS", "batch_size", "batch size", type=int)
    config.add_entry("dO", "dropout", "drop out rate", type=float)
    config.add_entry("wD", "weight_decay", "weight decay rate", type=float)
    config.add_entry("yS", "y_start", "", type=int)
    config.add_entry("sL", "seq_length", "number of past observations used for the prediction", type=int)
    config.add_entry("nH", "nhid", "number of hidden layers", type=int)
    config.add_entry("sv", "save",
                     "global dir for saving epochs result: specify model and dataset")  # save+'/Graph_WaveNet/METR-LA/'
    config.add_entry("aD", "adjdata", "global dir for sensor graph: specify dataset")  # adjdata+'/METR-LA/adj_mx.pkl'
    config.add_entry("oD", "output_dir",
                     "global dir for train, val, test data: specify the dataset")  # output_dir+'/METR-LA'
    config.add_entry("iD", "input_dir", "global dir for raw dataset files: specify dataset")  # input_dir+'/metr-la.h5'
    config.add_entry("aT", "adjtype", "")
    config.add_entry("pE", "print_every", "", type=int)
    config.add_entry("dv", "device", "")
    config.add_entry("expid", "expid", "", type=int)
    config.add_entry("oH", "output_horizon", "number of predicted time horizons", type=int)

    config.add_model_entry("gcn_bool", "whether to add graph convolution layer", action='store_true')
    config.add_model_entry("aptonly", "whether only adaptive adj", action='store_true')
    config.add_model_entry("addaptadj", "whether add adaptive adj", action='store_true')
    config.add_model_entry("randomadj", "whether random initialize adaptive adj", action='store_true')
    # config.add_model_entry("dow", "")

    config.add_entry("sn", "snorm", "1 if we want the spatial normalization", type=int)
    config.add_entry("tn", "tnorm", "1 if we want the  temporal normalization", type=int)
    config.add_entry("nD", "new_data", "1 if we want to generate new training data", type=int)
    config.add_entry("sH", "start_hour", "day time from which we start taking samples", type=int)
    config.add_entry("eH", "end_hour", "day time before which we stop taking samples", type=int)

    # GTS
    config.add_model_entry("use_cpu_only", "True if we want to use only CPU", action='store_true')
    config.add_entry("tp","temperature", "temperature value for gumbel-softmax.", type=float)
    config.add_entry("eUR", "epoch_use_regularization", " number of epochs after which we start regularization", type=int)
    config.add_entry("opt", "optimizer", "optimizer")
    config.add_entry("nS", "num_sample", "number of samples", type=int)

    # our model
    config.add_entry("lbE", "label_epochs", "number of epochs for training concept classification", type=int)
    config.add_entry("nhE", "nhid_encoder", "number of hidden layers for encoder in Transformer", type=int)
    config.add_entry("nhD", "nhid_decoder", "number of hidden layers for decoder in Transformer", type=int)
    config.add_entry("lOH", "label_output_horizon", "number of output horizons generated during concept classification", type=int)
    config.add_entry("lbEL", "label_epoch_limit", "up to this number, we use mse as loss then and after we use rmse in concept classification", type=int)
    config.add_entry("lbSPR", "label_save_pred_real", "1 if we want to save pred and real of concept classification", type=int)
    config.add_entry("lbSL", "label_save_loss", "1 if we want to save loss of concept classification", type=int)
    config.add_entry("pCC", "perform_concept_classification", "1 if we want to perform concept classification step", type=int)
    config.add_entry("pO", "perform_orchestration", "1 if we want to perform orchestration and final computation steps", type=int)
    config.add_entry("gP", "generate_predictors", "1 if we want to compute results of predictors", type=int)

    config.parse()

    db = PostgreDB(config._options)
    config.setup_db(db)


def setup_evaluation(evaluation):

    # speed
    evaluation.add_metric("weights", str, result_type="speed")
    evaluation.add_metric("mae", float)
    evaluation.add_metric("mape", float)
    evaluation.add_metric("rmse", float)
    evaluation.add_metric("bestid", int)
    evaluation.add_metric("time_spent", float)

    # label
    evaluation.add_metric("l_mae", float, result_type="label")
    evaluation.add_metric("l_mape", float, result_type="label")
    evaluation.add_metric("l_rmse", float, result_type="label")
    evaluation.add_metric("l_bestid", int, result_type="label")
    evaluation.add_metric("l_time_spent", float, result_type="label")

    # classification
    num_concepts = 4
    for i in range(num_concepts):
        evaluation.add_metric(f"precision_{i}", float, result_type="classification")
        evaluation.add_metric(f"recall_{i}", float, result_type="classification")
        evaluation.add_metric(f"f_score_{i}", float, result_type="classification")
        for j in range(num_concepts):
            evaluation.add_metric(f"cm_{i}{j}", int, result_type="classification")


class PredictorThread(threading.Thread):
    def __init__(self, threadID, name, config):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c = config

    def run(self):
        if not self.c.generate_predictors:
            return
        print("Starting " + self.name)

        # perform training
        self.c.id = register_run(self.c)
        # print(self.c)

        print('Beginning training ...')
        metrics, bestid, time_spent = train.run(self.c)  # metrics: mae, mape, rmse
        print('End training')

        # prepare evaluation
        evaluation = Evaluation(self.c)
        setup_evaluation(evaluation)

        # get metrics
        evaluation.compute_metric(metrics, bestid, time_spent)
        
        # save results into the data base
        print('Storing of the results ...')
        evaluation.store_results()
        print('results stored successfully')



if __name__ == "__main__":

    config = Configuration()
    setup_config(config)

    # define primary key: list of parameters common to all predictors
    primary_key = c_0 = config.get_configs()[0]

    # select configurations with different datasets
    dataset_list = []
    config_list = []
    for conf in config.get_configs():
        if conf.dataset_name not in dataset_list:
            dataset_list.append(conf.dataset_name)
            config_list.append(conf)

    # manage case of only data preparation
    if c_0.only_data_preparation:
        print('Data preparation ...')
        for conf in config_list:
            generate_training_data.run(conf)
    else:
        # manage case of new data
        root = u.get_root()
        training_data_path = f'{root}{c_0.output_dir}/{c_0.dataset_name.upper()}'

        if not os.path.exists(training_data_path) or c_0.new_data:  # verify the presence of training data
            print('Data preparation ...')
            for conf in config_list:
                generate_training_data.run(conf)

        predictors = []

        i = 1
        # create one thread for each c
        for c in config.get_configs():
            print(f'Starting experiment number {i} ...')
            # Create new thread
            predictor = PredictorThread(i, c.model_name, c)
            # Start new Threads
            predictor.start()
            # Add predictors to predictor list
            predictors.append(predictor)

            i += 1

        # launch "Our model" which is the main thread
        primary_key.datasets = dataset_list
        om_train.run(predictors, primary_key)

    print("End")






