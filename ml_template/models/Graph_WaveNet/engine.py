import torch.optim as optim
from models.Graph_WaveNet.model import *
import models.Graph_WaveNet.util as util


class trainer():
    def __init__(self, scaler, in_dim, output_horizon, num_nodes, nhid, dropout, lrate, wdecay, device, supports,
                 gcn_bool, addaptadj, aptinit, tnorm_bool=False, snorm_bool=False):
        self.model = gwnet(device, num_nodes, dropout, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool, supports=supports,
                           gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim,
                           out_dim=output_horizon, residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        # input.shape = [batch_size, 2, num_nodes, 12]
        # real_value.shape = [batch_size, num_nodes, 12]
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, embs = self.model(input)
        output = output.transpose(1, 3)

        # output.shape = [batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val, dim=1)
        # real.shape = [batch_size,1,num_nodes,12]
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val, params=None):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output, _ = self.model(input)
        output = output.transpose(1, 3)
        # output.shape = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0, params=params)
        mape = util.masked_mape(predict, real, 0.0, params=params).item()
        rmse = util.masked_rmse(predict, real, 0.0, params=params).item()
        return loss.item(), mape, rmse
