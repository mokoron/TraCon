import torch
import torch.nn as nn
import numpy as np
from models.Graph_WaveNet import util as gwn_util
from models.Our_model.engine import trainer
from models.Our_model import util
from models import util as models_util
import os
import util as u
import threading
from evaluation import Evaluation
from configuration import register_run
from main import setup_evaluation
import time

our_model = "Our_model"
classification = "Classification"
concepts = ['normal', 'jam', 'transition']

concept_classifier_task = "concept_classifier"
orchestration_final_task = "orchestration_final_computation"

lock = threading.Lock()


class ConceptClassifierThread(threading.Thread):
    def __init__(self, config, models, name):
        threading.Thread.__init__(self)
        with lock:
            self.our_config = config
            self.our_config.dataset_name = name
            self.models = models

    def run(self):
        # CONCEPT CLASSIFIER
        with lock:
            print("ACQUIRE LOCK")
            # complete arguments of config for concept classifier

            ## load data
            root = u.get_root()
            data_dir = f'{root}{self.our_config.output_dir}/{self.our_config.dataset_name.upper()}'
            dataloader = gwn_util.load_dataset(
                data_dir, self.our_config.batch_size, self.our_config.batch_size, self.our_config.batch_size)

            ## Get Shape of Data
            self.our_config.train_size, self.our_config.in_horizon, self.our_config.num_nodes, _ = dataloader[
                'x_train'].shape
            _, self.our_config.num_concepts, self.our_config.out_horizon, _ = dataloader['c_train'].shape
            print(f"dataset: {self.our_config.dataset_name} - train size N: {self.our_config.train_size}")
            # print("input horizon S: " + str(self.our_config.in_horizon))
            print(f"dataset: {self.our_config.dataset_name} - num_nodes E: {self.our_config.num_nodes}")
            # print("output horizon T: " + str(self.our_config.out_horizon))
            # print("number of concepts: " + str(self.our_config.num_concepts))

            self.our_config.dataloader = dataloader

            # label_results=[mae, mape, rmse], classification_results=[conf_matrix, precision, recall, f_score]
            label_results, classification_results, bestid, time_spent = concept_classifier(self.our_config)

            print("ABOUT TO RELEASE LOCK")


class OrchestrationFinalThread(threading.Thread):
    def __init__(self, config, models, name):
        threading.Thread.__init__(self)

        self.our_config = config
        self.our_config.dataset_name = name
        self.models = models
        self.our_config.models = models

    def run(self):
        # ORCHESTRATION AND FINAL COMPUTATION
        with lock:
            print("ACQUIRE LOCK")

            # complete arguments of config for orchestration and final computation

            ## load data
            root = u.get_root()
            data_dir = f'{root}{self.our_config.output_dir}/{self.our_config.dataset_name.upper()}'
            dataloader = gwn_util.load_dataset(
                data_dir, self.our_config.batch_size, self.our_config.batch_size, self.our_config.batch_size)

            ## Get Shape of Data
            self.our_config.train_size, self.our_config.in_horizon, self.our_config.num_nodes, _ = dataloader[
                'x_train'].shape
            _, self.our_config.num_concepts, self.our_config.out_horizon, _ = dataloader['c_train'].shape
            print(f"dataset: {self.our_config.dataset_name} - train size N: {self.our_config.train_size}")
            # print("input horizon S: " + str(self.our_config.in_horizon))
            print(f"dataset: {self.our_config.dataset_name} - num_nodes E: {self.our_config.num_nodes}")
            # print("output horizon T: " + str(self.our_config.out_horizon))
            # print("number of concepts: " + str(self.our_config.num_concepts))

            self.our_config.dataloader = dataloader

            t1 = time.time()
            concept_weights_pairs = orchestration(self.models,
                                                  self.our_config)  # pairs.shape = (num_concepts, num_models)

            # final computation with ablation study
            ablation_list = [concept_weights_pairs]
            for index in range(len(ablation_list)):
                ablation = f"-{concepts[index]}" if index > 0 else ""
                model = f"{our_model}{ablation}"
                pairs = ablation_list[index]

                # transform weights into a dictionary, then casted into string. Keys are model names, values are weight vector
                weights = str({self.models[i]: pairs[:, i].tolist() for i in range(len(self.models))})
                # replace ' and " with empty string
                weights = weights.replace("'", "").replace('"', "")

                final_pred, final_real = final_computation(self.models, self.our_config,
                                                           pairs)  # shape = (labels_horizons, ~num_sample*num_nodes)

                # compute metrics
                amae = []
                amape = []
                armse = []
                for i in range(self.our_config.label_output_horizon):
                    pred = final_pred[i, :]
                    real = final_real[i, :]
                    params = {"phase": "test", "model_name": model, "dataset_name": self.our_config.dataset_name,
                              "horizon": i, "epoch": "", "save_loss": self.our_config.save_loss,
                              "save_pred_real": self.our_config.save_pred_real}
                    metrics = gwn_util.metric(pred, real, params=params)
                    log = '{} - dataset: {} - horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                    print(log.format(model, self.our_config.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])

                    # prepare evaluation
                    self.our_config.model_name = model
                    self.our_config.id = register_run(self.our_config)
                    our_evaluation = Evaluation(self.our_config)
                    setup_evaluation(our_evaluation)

                    # save results inside the database. bestid here is substituted by the horizon
                    our_evaluation.compute_metric(metrics, i+1, 0.0, weights)

                    print(f'Storing of the results: {model} - dataset: {self.our_config.dataset_name} - horizon: {i+1}...')
                    our_evaluation.store_results()
                    print(f'results stored successfully: {model} - dataset: {self.our_config.dataset_name} - horizon: {i+1}')

                mean_amae, mean_amape, mean_armse = np.mean(amae), np.mean(amape), np.mean(armse)
                log = '{} - dataset: {} - On average over the {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                print(log.format(model, self.our_config.dataset_name, self.our_config.label_output_horizon, mean_amae,
                                 mean_amape, mean_armse))

                t2 = time.time()
                metrics = [mean_amae, mean_amape, mean_armse]
                time_spent = t2 - t1

                # prepare evaluation
                self.our_config.model_name = model
                self.our_config.id = register_run(self.our_config)
                our_evaluation = Evaluation(self.our_config)
                setup_evaluation(our_evaluation)

                # save results inside the database ()
                our_evaluation.compute_metric(metrics, 0, time_spent, weights)

                print(f'Storing of the results: {model} - dataset: {self.our_config.dataset_name} ...')
                our_evaluation.store_results()
                print(f'results stored successfully: {model} - dataset: {self.our_config.dataset_name}')

            print("ABOUT TO RELEASE LOCK")


def run(predictors, config):
    print("Starting Main Thread")

    # get the list of available trained models
    models = [predictor.c.model_name for predictor in predictors]
    models = list(dict.fromkeys(models))  # without duplicates

    # concept classifier, classification result is a dictionary wit keys concept name
    if config.perform_concept_classification:
        perform_task(config, models, concept_classifier_task)

    # Wait for all other predictors (predictors) to complete
    try:
        for predictor in predictors:
            predictor.join()
    except Exception as e:
        print(e.args[0])
    print("Synchronized predictors with concept classifier, we can start orchestration")

    # orchestration and final computation
    if config.perform_orchestration:
        time.sleep(1)  # wait a little bit for all data to be stores in memory
        perform_task(config, models, orchestration_final_task)


def perform_task(config, models, task_name):
    print(f"Beginning task: {task_name}")
    classes = {
        f"{concept_classifier_task}": ConceptClassifierThread,
        f"{orchestration_final_task}": OrchestrationFinalThread
    }
    for dataset in config.datasets:
        threads = []
        # Create new thread
        thread = classes[task_name](config, models, dataset)  # config, models, name, our_config
        # Start new Threads
        thread.start()
        # Add predictors to predictor list
        threads.append(thread)

        for thread in threads:
            thread.join()
    print(f"Completed task: {task_name}")


def training_validation(our_config):
    device = torch.device(our_config.device) if torch.cuda.is_available() else torch.device('cpu')
    scaler = our_config.dataloader['scaler']

    engine = trainer(E=our_config.num_nodes, device=device, scaler=scaler, num_encoder_layers=our_config.nhid_encoder,
                     num_decoder_layers=our_config.nhid_decoder, epoch_limit=our_config.label_epoch_limit)

    root = u.get_root()
    save_dir = f'{root}{our_config.save}/{classification}/{our_config.dataset_name.upper()}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saving directory at ' + save_dir)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_pred = None
    t1 = t2 = 0.0
    val_real = []
    for i in range(1, our_config.label_epochs + 1):
        train_loss = []
        train_mape = []
        train_mae = []
        t1 = time.time()
        our_config.dataloader['train_loader'].shuffle()
        for iter, (x, _, y) in enumerate(our_config.dataloader['train_loader'].get_iterator()):
            # x_shape = (batch_size, horizon, num_nodes)
            # y_shape = (batch_size, num_concepts, horizon, num_nodes)
            trainx = torch.Tensor(x).to(device)
            trainy = torch.Tensor(y).to(device)
            input, real_val = util.prepare_in(trainx, trainy)

            metrics = engine.train(input, real_val, i, trainy.shape)

            train_mae.append(metrics[0])
            train_mape.append(metrics[1])
            train_loss.append(metrics[2])

            if iter % our_config.print_every == 0:
                log = '{} - dataset: {}. Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train MAE: {:.4f}'
                print(log.format(classification, our_config.dataset_name,
                                 iter, train_loss[-1], train_mape[-1], train_mae[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_loss = np.mean(train_loss)

        # validation
        phase = "val"

        s1 = time.time()
        [valid_mae, valid_mape, valid_loss], val_preds, val_real = util.get_metrics(our_config, engine, phase, ep=i,
                                                                                    name=classification)
        s2 = time.time()
        log = '{} - dataset: {}. Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(classification, our_config.dataset_name, i, (s2 - s1)))
        val_time.append(s2 - s1)

        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_loss = np.mean(valid_loss)

        his_loss.append(mvalid_loss)

        log = '{} - dataset: {}. Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train MAE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid MAE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(classification, our_config.dataset_name, i, mtrain_loss, mtrain_mape, mtrain_mae,
                         mvalid_loss, mvalid_mape, mvalid_mae, (t2 - t1)), flush=True)
        # save only if less than the minimum
        if mvalid_loss == np.min(his_loss):
            best_pred = val_preds
            torch.save(engine.transformer_model.state_dict(),
                       save_dir + "_exp" + str(our_config.expid) + "_best_" + ".pth")
            # remove all folders different from the best one
            models_util.clean_directory(config=our_config, index=i, phase=phase, name=classification)
        print(
            "{} - dataset: {}. Average Training Time: {:.4f} secs/epoch".format(classification, our_config.dataset_name,
                                                                                np.mean(train_time)))
        print("{} - dataset: {}. Average Inference Time: {:.4f} secs".format(classification, our_config.dataset_name,
                                                                             np.mean(val_time)))

    bestid = np.argmin(his_loss)
    print("{} - dataset: {}. Training finished".format(classification, our_config.dataset_name))
    print("{} - dataset: {}. The valid loss on best model is {}".format(classification, our_config.dataset_name,
                                                                        round(his_loss[bestid], 4)))
    models_util.clean_directory(config=our_config, index=bestid + 1, phase="val", rename=True, name=classification)

    # print classification results
    for i in range(len(concepts)):
        print("\nConcept: ", concepts[i])
        util.get_classification_stats(best_pred[i], val_real[i], device=device)

    # test
    phase = "test"
    engine.transformer_model.load_state_dict(torch.load(
        save_dir + "_exp" + str(our_config.expid) + "_best_" + ".pth"))

    [amae, amape, armse], preds, real = util.get_metrics(config=our_config, engine=engine, phase=phase,
                                                         name=classification)
    mean_amae, mean_amape, mean_armse = np.mean(amae), np.mean(amape), np.mean(armse)

    log = '{} - dataset: {}. On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'

    print(log.format(classification, our_config.dataset_name, our_config.label_output_horizon, np.mean(
        amae), np.mean(amape), np.mean(armse)))

    multi_preds = util.get_multiclass(preds, device=device)
    multi_real = util.get_multiclass(real, device=device)
    matrix, precision, recall, f1_score = util.get_classification_stats(multi_preds, multi_real, device=device)

    label_results = mean_amae, mean_amape, mean_armse
    classification_results = matrix, precision, recall, f1_score

    return label_results, classification_results, bestid, (t2 - t1)


def concept_classifier(our_config):
    # training, validation and test
    result = training_validation(our_config)
    return result


def orchestration(models, our_config):  # remove index in case of ablation study
    device = torch.device(our_config.device) if torch.cuda.is_available() else torch.device('cpu')
    phase = "val"
    # load validation labels from concept classifier
    params = {"phase": phase, "model_name": classification, "dataset_name": our_config.dataset_name,
              "epoch": "epoch_0/"}
    val_labels = get_tensor(our_config, params).to(device)  # shape = (horizons, num_samples, num_concepts, num_nodes)

    labels_horizons, num_samples, num_concepts, num_nodes = val_labels.shape

    # load validation results from predictors
    val_real = []
    val_preds = []
    for model in models:
        params = {"phase": phase, "model_name": model, "dataset_name": our_config.dataset_name, "epoch": "epoch_0/"}
        model_real = get_tensor(our_config, params, "real").to(device)
        model_pred = get_tensor(our_config, params).to(device)  # shape = (horizons, num_samples, num_nodes)
        val_real.append(model_real.unsqueeze(dim=-1))
        val_preds.append(model_pred.unsqueeze(dim=-1))
    val_real = torch.cat(val_real, dim=-1)
    val_preds = torch.cat(val_preds, dim=-1)  # shape = (horizons, num_samples, num_nodes, num_models)

    preds_horizons, _, _, num_models = val_preds.shape

    pairs = []
    # for each concept, perform the orchestration logic
    val_preds = val_preds[:labels_horizons]
    val_real = val_real[:labels_horizons]
    for i in range(num_concepts):
        out = get_orchestration_output(val_preds, val_real, val_labels[:, :, i, :],
                                       device=device)  # out.shape = (num_models)
        pairs.append(out.unsqueeze(dim=0))
    pairs = torch.cat(pairs, dim=0)  # pairs.shape = (num_concepts, num_models)

    return pairs


def final_computation(models, our_config, pairs):
    device = torch.device(our_config.device) if torch.cuda.is_available() else torch.device('cpu')
    phase = "test"
    # load test labels from concept classifier
    params = {"phase": phase, "model_name": classification, "dataset_name": our_config.dataset_name, "epoch": ""}
    test_labels = get_tensor(our_config, params)  # shape = (horizons, num_samples, num_concepts, num_nodes)

    labels_horizons, num_samples, num_concepts, num_nodes = test_labels.shape

    # load test results from predictors
    test_real = []
    test_preds = []
    for model in models:
        params = {"phase": phase, "model_name": model, "dataset_name": our_config.dataset_name, "epoch": ""}
        model_real = get_tensor(our_config, params, "real").to(device)
        model_pred = get_tensor(our_config, params)
        test_real.append(model_real.unsqueeze(dim=0))
        test_preds.append(model_pred.unsqueeze(dim=0))

        # save results of each model in the database
        our_config.model_name = model
        our_config.id = register_run(our_config)
        our_evaluation = Evaluation(our_config)
        setup_evaluation(our_evaluation)
        metrics = gwn_util.metric(model_pred, model_real)
        our_evaluation.compute_metric(metrics)
        our_evaluation.store_results()

        # print result by time horizon
        for i in range(our_config.output_horizon):
            pred = model_pred[i, :, :]
            real = model_real[i, :, :]
            metrics = gwn_util.metric(pred, real)
            if phase == "test":
                log = '{} - dataset: {}. Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                print(log.format(our_config.model_name, our_config.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))

    test_real = torch.cat(test_real, dim=0)
    test_preds = torch.cat(test_preds, dim=0)  # shape = (num_models, horizons, num_samples, num_nodes)

    test_preds = test_preds[:, :labels_horizons]
    test_real = test_real[:, :labels_horizons]

    # compute prediction using weights --> list of lists of tensors [labels_horizons][num_concepts]
    f_preds, f_real = compute_prediction(test_preds, test_real, test_labels, pairs, labels_horizons, our_config=our_config)

    final_preds = []
    final_real = []
    for horizon in range(labels_horizons):
        p = torch.tensor([]).to(device)
        r = torch.tensor([]).to(device)
        for concept in range(num_concepts):
            p = torch.cat([p, f_preds[horizon][concept]])
            r = torch.cat([r, f_real[horizon][concept]])

        final_preds.append(p.unsqueeze(dim=0))
        final_real.append(r.unsqueeze(dim=0))

    # make sure that all sensors in final_preds and final_real are of the same dimension
    final_preds = verify_list(final_preds)
    final_real = verify_list(final_real)

    final_preds = torch.cat(final_preds)
    final_real = torch.cat(final_real)  # shape = (labels_horizons, ~num_sample*num_nodes)
    min_value = np.min([final_preds.shape[1], final_real.shape[1]])

    return final_preds[:, :min_value], final_real[:, :min_value]


def get_tensor(our_config, params, partition_id="pred"):
    horizons = []
    for i in range(our_config.label_output_horizon):
        params["horizon"] = i
        horizon = models_util.load_element(params, "pred_real",
                                           partition_id)  # shape = (num_samples, num_concepts, num_nodes) OR (num_samples, num_nodes)
        horizons.append(horizon.unsqueeze(dim=0))
    horizons = torch.cat(horizons, dim=0)
    return horizons  # shape = (horizons, num_samples, num_concepts, num_nodes) OR (horizons, num_samples, num_nodes)


def get_orchestration_output(val_preds, val_real, val_labels, device):
    # val_preds.shape = (labels_horizons, num_samples, num_nodes, num_models)
    # val_real.shape = (labels_horizons, num_samples, num_nodes)
    # val_labels.shape = (labels_horizons, num_samples, num_nodes)

    # construct val_labels in shape = (num_samples, labels_horizon, num_nodes, num_models)
    labels_horizons, num_samples, num_nodes, num_models = val_preds.shape
    val_labels = val_labels.repeat(num_models, 1, 1).reshape(num_models, labels_horizons, num_samples, num_nodes)
    val_labels = val_labels.permute(1, 2, 3, 0)

    zero = torch.zeros(val_real.shape).to(device)
    one = torch.ones(val_real.shape).to(device)

    errors = abs(val_real - val_preds).float()

    # filter the error according to val_labels
    errors = torch.where(val_labels > 0, errors, zero)

    z = torch.min(errors, dim=-1).values
    min_error = z.repeat(num_models, 1, 1, 1).permute(1, 2, 3, 0)

    x = errors.reshape(labels_horizons * num_samples * num_nodes, num_models).float()
    y = torch.where((errors == min_error), one, zero).reshape(labels_horizons * num_samples * num_nodes,
                                                              num_models).float()
    z = z.reshape(labels_horizons * num_samples * num_nodes).float()


    # data filtering
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(device)
    q_0, q_1, q_2, q_3, q_4 = torch.quantile(z, q)
    low = q_0
    high = q_4
    x = x[(z >= low) & (z <= high), :]
    y = y[(z >= low) & (z <= high), :]
    z = z[(z >= low) & (z <= high)]

    # split data into train and test
    s = x.shape[0]
    num_train = int(0.8 * s)
    num_test = s - num_train

    x_train, y_train, z_train = x[:num_train], y[:num_train], z[:num_train]
    x_test, y_test, z_test = x[-num_test:], y[-num_test:], z[-num_test:]

    # train a classifier (Linear layer = FCL)
    in_features = x.shape[1]
    out_features = y.shape[1]
    eps = 10
    ##  initialize weights
    weight = torch.randn(out_features, in_features).float().to(device)

    layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False, device=device)
    layer.weight = nn.Parameter(weight)
    ## train
    for ep in range(eps):
        layer = train(layer, x_train, y_train, device=device)
    ## test
    output_test = test(layer, x_test)    

    # mean probabilities for each model (Reduce dimension)
    output_mean = output_test.mean(dim=0)

    output_mean = output_mean + x.mean(dim=0)

    # revert the result so that the biggest weight goes to the one with lowest error.
    output_mean = torch.sum(output_mean)*torch.ones(output_mean.shape) - output_mean

    # softmax layer to get weights
    output = util.rescale(output_mean)

    return output


def train(layer, input, real_val, device):
    layer.train()
    # Fully Connected Layer
    output = layer(input)
    # Softmax activation function
    output = util.rescale(output, dim=1)  # dim = 1 means sum of all columns for each row is 1
    loss = util.masked_bce(output, real_val, device=device)
    loss.backward()
    return layer


def test(layer, input):
    output = layer(input)
    output = util.rescale(output, dim=1)
    return output


def transform_weights(pairs, concepts=0):  # remove could be a list
    # puts weights of normal concept at the position of the missing concepts
    # pairs.shape = (num_concepts, num_models)
    # output. shape = (num_concepts+len(remove), num_nodes)

    pointer = 0
    if concepts == pointer:
        return pairs

    # order remove
    remove = np.sort(concepts) if isinstance(concepts, type([])) else [concepts]

    # substitute values
    output = pairs
    normal = pairs[[pointer]]
    for i in range(len(remove), 0, -1):
        output = torch.cat([output[:remove[i - 1]], normal, output[remove[i - 1] + 1:]])

    return output  # shape = pairs.shape


def compute_prediction(preds, real, labels, pairs, labels_horizons, our_config):
    device = torch.device(our_config.device) if torch.cuda.is_available() else torch.device('cpu')
    # preds.shape = (num_models, horizons, num_samples, num_nodes)
    # labels.shape = (horizons, num_samples, num_concepts, num_nodes)
    # f_preds = list of lists of tensors [labels_horizons][num_concepts]
    # f_real = list of lists of tensors [labels_horizons][num_concepts]

    ones = torch.ones(labels.shape)
    labels = torch.where(labels > 0, ones, labels)

    num_concepts = pairs.shape[0]
    num_models = preds.shape[0]

    f_preds, f_real = [], []
    for horizon in range(labels_horizons):
        r = []
        p = []
        for concept in range(num_concepts):
            r_c = [torch.tensor(real[model, horizon, labels[horizon, :, concept, :] > 0]).to(device) for model in
                   range(num_models)]
            p_c = [torch.tensor(preds[model, horizon, labels[horizon, :, concept, :] > 0]).to(device) for model in
                   range(num_models)]
            # compute weighted sum
            r_c = models_util.weighted_sum(pairs[concept], r_c, type_="linear")
            p_c = models_util.weighted_sum(pairs[concept], p_c, type_="linear")
            r.append(r_c)
            p.append(p_c)
        f_preds.append(p)
        f_real.append(r)

    return f_preds, f_real


def verify_list(input_list):
    # input_list is a list of tensors of dimension [labels_horizons, ~num_samples*num_nodes]
    # output_list is list of tensors
    min_dim = np.min([i.shape[1] for i in input_list])
    output_list = [input_list[i][:, :min_dim] for i in range(len(input_list))]

    return output_list