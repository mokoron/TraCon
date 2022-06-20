import torch
from torchmetrics import ConfusionMatrix, Precision, Recall, F1Score
from models.Graph_WaveNet import util as gwn_util
import torch.nn as nn
import numpy as np


def rescale(input, dim=0):
    # put all value in range [0,1], the sum along dimension dim is 1.

    softmax = torch.nn.Softmax(dim=dim)
    output = softmax(input)
    return output


def binary_mapping(input, device, limit=0.5):  # 0.25 is the logic limit (always have at least 1 concept)
    # limit is the probability under which the value is mapped to 0
    zero = torch.zeros(input.shape).to(device)
    one = torch.ones(input.shape).to(device)
    output = torch.where(input <= limit, zero, one)

    return output   # shape = input.shape


def disambiguate(input, device):
    # tranforms input with overlaps with output without overlaps (num_concpts, num_sample, num_nodes)
    # output.shape = input.shape =  (num_samples, num_concepts, out_horizon, num_nodes)

    num_samples, num_concepts, out_horizon, num_nodes = input.shape
    zero = torch.zeros(num_samples, out_horizon, num_nodes).to(device)

    output = []
    for i in range(num_concepts):
        sum_stronger_i = zero
        for j in range(i+1, num_concepts):
            sum_stronger_i += input[:, j, :, :]
        c_i = output = torch.where(sum_stronger_i >= 1, zero, input[:, i, :, :])
        output.append(c_i)
    output = torch.stack(output, dim=1)

    return output


def confusion_matrix(y_pred, y_true, device):
    # cast tensors to integer to be able to use average != 'micro'
    y_pred, y_true = y_pred.int(), y_true.int()
    classes = torch.unique(torch.cat([y_pred, y_true]))
    num_classes = len(classes)

    confmat = ConfusionMatrix(num_classes=num_classes).to(device)
    matrix = confmat(y_pred, y_true)

    return matrix  # 0: real, 1: pred. Classes ordered like in "classes"


def precision_recall_fscore(y_pred, y_true, device):
    # cast tensors to integer to be able to use average != 'micro'
    y_pred, y_true = y_pred.int(), y_true.int()
    classes = torch.unique(torch.cat([y_pred, y_true]))
    num_classes = len(classes)
    p = Precision(num_classes=num_classes, average=None, mdmc_average='global').to(device)
    r = Recall(num_classes=num_classes, average=None, mdmc_average='global').to(device)
    f1 = F1Score(num_classes=num_classes, average=None, mdmc_average='global').to(device)

    precision, recall, f1_score = p(y_pred, y_true), r(y_pred, y_true), f1(y_pred, y_true)

    return precision, recall, f1_score


def get_classification_stats(pred, real, device, limit=None):
    # to visualize the confusion matric, precision, recall and f1 score

    if limit is not None:
        # we transform y_pred into binary format
        pred = binary_mapping(pred, device, limit=limit)

    matrix = confusion_matrix(pred, real, device)
    precision, recall, f1_score = precision_recall_fscore(pred, real, device)

    print("confusion matrix: \n", matrix)
    print("precision: ", precision, "\nrecall: ", recall, "\nf1 score: ", f1_score)
    return matrix, precision, recall, f1_score


def get_multiclass(input, device):
    # tranforms input with 0,1 into input with 0,1,2,3,... (num_concpts, num_sample, num_nodes)
    # --> (num_sample, num_nodes)

    num_samples, num_concepts, out_horizon, num_nodes = input.shape
    zero = torch.zeros(num_samples, out_horizon, num_nodes).to(device)

    output = zero
    for i in range(num_concepts):
        sum_stronger_i = zero
        for j in range(i + 1, num_concepts):
            sum_stronger_i += input[:, j, :, :]
        c_i = output = torch.where(sum_stronger_i >= 1, zero, input[:, i, :, :])
        output += (i+1)*c_i

    return output   # values between 1 and num_concepts (included)


def get_metrics(config, engine, phase, ep=None, name=None):
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device('cpu')
    epoch = {"val": f"epoch_{ep}/", "test": ""}
    model_name = name if name is not None else config.model_name
    outputs = []
    realy = torch.Tensor(config.dataloader[f'c_{phase}']).to(device)     # shape = (num_samples, num_concepts, horizon, num_nodes)

    for iter, (x, _, y) in enumerate(config.dataloader[f'{phase}_loader'].get_iterator()):
        # x_shape = (batch_size, horizon, num_nodes, input_size)
        # y_shape = (batch_size, num_concepts, horizon, num_nodes)
        phasex = torch.Tensor(x).to(device)  # phasex to say valx or testx
        phasey = torch.Tensor(y).to(device)
        # phasex, phasey = phasex.transpose(1, 0), phasey.transpose(1, 0)
        with torch.no_grad():
            input, real_val = prepare_in(phasex, phasey)
            preds = engine.eval_test(input, real_val, phasey.shape)    # shape = (batch_size, num_concepts, horizon, num_nodes)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)  # 0 is dimension for batch_size
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(config.label_output_horizon):
        pred = yhat[:, :, i, :]
        real = realy[:, :, i, :]
        params = {"phase": phase, "model_name": model_name, "dataset_name": config.dataset_name, "horizon": i,
                  "epoch": epoch[phase], "save_loss": config.label_save_loss, "save_pred_real": config.label_save_pred_real}
        metrics = gwn_util.metric(pred, real, params=params)
        if phase == "test":
            log = '{} - dataset: {}. Evaluate best model on test data for horizon {:d}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test MAE: {:.4f}'
            print(log.format(model_name, config.dataset_name, i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    return [amae, amape, armse], yhat, realy


def prepare_in(phasex, phasey):
    # phasex.shape = (batch_size, in_horizon, num_nodes, input_size)
    # phasey.shape = (batch_size, num_concepts, out_horizon, num_nodes)
    # input_.shape = [in_horizon, batch_size, num_nodes]
    # real_val.shape = [num_concepts*out_horizon, batch_size, num_nodes]
    s = phasey.shape

    input_ = phasex[:, :, :, 0].permute(1, 0, 2)
    real_val = phasey.permute(1, 2, 0, 3).reshape(s[1] * s[2], s[0], s[3])
    return input_, real_val


def prepare_out(preds, s):
    # preds.shape = [num_concepts*out_horizon, batch_size, num_nodes]
    # out.shape = (batch_size, num_concepts, out_horizon, num_nodes)
    out = preds.reshape(s[1], s[2], s[0], s[3]).permute(2, 0, 1, 3)
    return out


def masked_bce(preds, labels, null_val=np.nan, device=torch.device("cpu")):
    preds = torch.tensor(preds, requires_grad=True).to(device)
    labels = torch.tensor(labels, requires_grad=True).to(device)
    loss = nn.BCELoss()
    bce_loss = loss(preds, labels)
    bce_loss = output = torch.tensor(bce_loss.item(), requires_grad=True).to(device)

    return bce_loss
