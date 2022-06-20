import torch
import torch.optim as optim
import torch.nn as nn
from models.Graph_WaveNet import util as gwn_util
from models.Our_model import util


class trainer():
    def __init__(self, E, device, scaler, num_encoder_layers, num_decoder_layers, epoch_limit=10):
        self.transformer_model = nn.Transformer(d_model=E, nhead=E, device=device,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers)
        self.transformer_model.to(device)
        self.optimizer = optim.Adam(self.transformer_model.parameters())
        self.loss = gwn_util.masked_mse
        self.scaler = scaler
        self.clip = 5
        self.epoch_limit = epoch_limit
        self.device = device

    def train(self, input, real_val, epoch, shape):     # shape = (batch_size, num_concepts, horizon, num_nodes)
        if epoch == self.epoch_limit:
            self.loss = gwn_util.masked_rmse
        # input = [in_horizon=S, batch_size=N, num_nodes=E]
        # real_value = [num_concepts*out_horizon, batch_size, num_nodes]

        # input layer
        real = util.prepare_out(real_val, shape)    # real = (batch_size, num_concepts, out_horizon, num_nodes)
        self.transformer_model.train()

        # optimization layer
        self.optimizer.zero_grad()

        # prediction layer with transformer
        output = self.transformer_model(input, real_val)
        output = util.prepare_out(output, shape)    # output = (batch_size, num_concepts, out_horizon, num_nodes)

        # Relu layer
        predict = util.binary_mapping(output, device=self.device, limit=0)     # predict.shape = (batch_size, num_concepts, out_horizon, num_nodes)

        num_samples, _, num_nodes, num_horizon = predict.shape
        for r in range(num_nodes):
            for h in range(num_horizon):
                for s in range(num_samples):
                    predict[s, 0, r, h] = 1 if torch.sum(predict[s, :, r, h]) == 0 else predict[s, 0, r, h]

        predict = util.disambiguate(predict, self.device)
        predict = torch.tensor(predict, requires_grad=True).to(self.device)

        # backpropagation layer (weights update)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), self.clip)
        self.optimizer.step()

        # output layer
        mape = gwn_util.masked_mape(predict, real, 0.0).item()
        mae = gwn_util.masked_mae(predict, real, 0.0).item()
        if epoch >= self.epoch_limit:
            return mae, mape, loss.item()
        else:
            return mae, mape, torch.sqrt(loss).item()

    def eval(self, input, real_val, shape):
        # input = [in_horizon=S, batch_size=N, num_nodes=E]
        # real_value = [num_concepts*out_horizon, batch_size, num_nodes]
        real = util.prepare_out(real_val, shape)  # real = (batch_size, num_concepts, out_horizon, num_nodes)

        self.transformer_model.eval()
        output = self.transformer_model(input, real_val)
        output = util.prepare_out(output, shape)  # output = (batch_size, num_concepts, out_horizon, num_nodes)

        # Relu layer
        predict = util.binary_mapping(output, device=self.device, limit=0)

        num_samples, _, num_nodes, num_horizon = predict.shape
        for r in range(num_nodes):
            for h in range(num_horizon):
                for s in range(num_samples):
                    predict[s, 0, r, h] = 1 if torch.sum(predict[s, :, r, h]) ==  0 else predict[s, 0, r, h]

        # disambiguation layer
        predict = util.disambiguate(predict, self.device)
        rmse = self.loss(predict, real, 0.0).item()
        mape = gwn_util.masked_mape(predict, real, 0.0).item()
        mae = gwn_util.masked_mae(predict, real, 0.0).item()
        return [mae, mape, rmse], predict

    def eval_test(self, input, real_val, shape):
        # input = [in_horizon=S, batch_size=N, num_nodes=E]
        # real_value = [num_concepts*out_horizon, batch_size, num_nodes]

        output = self.transformer_model(input, real_val)
        output = util.prepare_out(output, shape)  # output = (batch_size, num_concepts, out_horizon, num_nodes)

        # Relu layer
        predict = util.binary_mapping(output, device=self.device, limit=0)

        num_samples, _, num_nodes, num_horizon = predict.shape
        for r in range(num_nodes):
            for h in range(num_horizon):
                for s in range(num_samples):
                    predict[s, 0, r, h] = 1 if torch.sum(predict[s, :, r, h]) ==  0 else predict[s, 0, r, h]

        predict = util.disambiguate(predict, self.device)
        return predict  # predict = (batch_size, num_concepts, out_horizon, num_nodes)
