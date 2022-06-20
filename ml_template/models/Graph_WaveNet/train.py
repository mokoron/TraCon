import torch
import numpy as np
import util as u
import time
from models import util as models_util
import os
from models.Graph_WaveNet.engine import trainer as Graph_WaveNet_trainer
from models.Graph_WaveNet import util as uGWN


def run(config):
    # define run specific parameters based on user input
    root = u.get_root()
    save_dir = f'{root}{config.save}/{config.model_name}/{config.dataset_name.upper()}/'
    adjdata_file = f'{root}{config.adjdata}/{config.dataset_name.upper()}/adj_mx.pkl'
    output_dir = f'{root}{config.output_dir}/{config.dataset_name.upper()}'

    # load data
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')

    if config.model_name.lower() == 'Graph_WaveNet'.lower():
        util = uGWN
        Trainer = Graph_WaveNet_trainer
        config.snorm = config.tnorm = 0
        print('Graph WaveNet trainer')
    elif config.model_name.lower() == 'ST_Norm'.lower():
        util = uGWN
        Trainer = Graph_WaveNet_trainer
        config.snorm = config.tnorm = 1
        print('ST_Norm trainer')
    else:
        raise AssertionError(f"Model {config.model_name} not a wavenet")

    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adjdata_file, config.adjtype)
    dataloader = util.load_dataset(output_dir, config.batch_size, config.batch_size, config.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print('Configuration done properly ...')

    if config.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if config.aptonly:
        supports = None

    # Get Shape of Data
    N, _, num_nodes, in_dim = dataloader['x_train'].shape  # _ is seq_length

    engine = Trainer(scaler, in_dim, config.output_horizon, num_nodes, config.nhid, config.dropout,
                     config.learning_rate, config.weight_decay, device, supports, config.gcn_bool, config.addaptadj,
                     adjinit, tnorm_bool=config.tnorm, snorm_bool=config.snorm)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saving directory at ' + save_dir)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, config.epochs + 1):
        # training

        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, _) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:, 0, :, :])  # 0: speed

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % config.print_every == 0:
                log = '{} - dataset: {}. Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(config.model_name, config.dataset_name, iter, train_loss[-1], train_mape[-1],
                                 train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        # validation
        phase = "val"

        s1 = time.time()
        valid_loss, valid_mape, valid_rmse = get_metrics(config=config, dataloader=dataloader, device=device,
                                                         engine=engine, scaler=scaler, phase=phase, util=util, ep=i)
        s2 = time.time()
        log = '{} - dataset: {}. Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(config.model_name, config.dataset_name, i, (s2 - s1)))
        val_time.append(s2 - s1)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)

        log = '{} - dataset: {}. Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(config.model_name, config.dataset_name, i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                         mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        # save only if less than the minimum
        if mvalid_loss == np.min(his_loss):
            torch.save(engine.model.state_dict(), save_dir + "_exp" + str(config.expid) + "_best_" + ".pth")
            # remove all folders different from the best one
            models_util.clean_directory(config=config, index=i, phase=phase)

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))
    models_util.clean_directory(config=config, index=bestid + 1, phase="val", rename=True)

    # testing
    phase = "test"
    engine.model.load_state_dict(torch.load(save_dir + "_exp" + str(config.expid) + "_best_" + ".pth"))

    amae, amape, armse = get_metrics(config=config, dataloader=dataloader, device=device, engine=engine, scaler=scaler,
                                     phase=phase, util=util)

    mean_amae = np.mean(amae)
    mean_amape = np.mean(amape)
    mean_armse = np.mean(armse)
    log = '{} - dataset: {}. On average over {} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(config.model_name, config.dataset_name, config.output_horizon, mean_amae, mean_amape, mean_armse))

    return [mean_amae, mean_amape, mean_armse], bestid


def get_metrics(config, dataloader, device, engine, scaler, phase, util, ep=None):
    epoch = {"val": f"epoch_{ep}/", "test": ""}
    outputs = []
    realy = torch.Tensor(dataloader[f'y_{phase}']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y, _) in enumerate(dataloader[f'{phase}_loader'].get_iterator()):
        phasex = torch.Tensor(x).to(device)  # phasex to say valx or testx
        phasex = phasex.transpose(1, 3)
        with torch.no_grad():
            preds, embs = engine.model(phasex)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    output_horizon = np.min([config.output_horizon, config.seq_length])
    for i in range(output_horizon):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        params = {"phase": phase, "model_name": config.model_name, "dataset_name": config.dataset_name, "horizon": i,
                  "epoch": epoch[phase], "save_loss": config.save_loss, "save_pred_real": config.save_pred_real}
        metrics = util.metric(pred, real, params=params)
        if phase == "test":
            log = '{} - dataset: {}.Best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(config.model_name, config.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    return amae, amape, armse
