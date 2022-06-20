import torch
from models import util as models_util


def masked_mae_loss(y_pred, y_true, params=None):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if params is not None:
        models_util.save_element(loss, params, "loss", "mae")
    return loss.mean()


def masked_mape_loss(y_pred, y_true, params=None):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if params is not None:
        models_util.save_element(loss, params, "loss", "mape")

        # construction of the prediction and real value in shape [num_time_steps, num_nodes]
        p = y_pred * mask
        r = y_true * mask
        p = torch.where(torch.isnan(p), torch.zeros_like(p), p)
        r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
        models_util.save_element(p, params, "pred_real", "pred")
        models_util.save_element(r, params, "pred_real", "real")
    return loss.mean()


def masked_rmse_loss(y_pred, y_true, params=None):
    return torch.sqrt(masked_mse_loss(y_pred, y_true, params=params))


def masked_mse_loss(y_pred, y_true, params=None):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    if params is not None:
        models_util.save_element(loss, params, "loss", "mse")
    return loss.mean()


def metric(pred, real, params=None):
    mae = masked_mae_loss(pred, real, params=params).item()
    mape = masked_mape_loss(pred, real, params=params).item()
    rmse = masked_rmse_loss(pred, real, params=params).item()
    return mae, mape, rmse



