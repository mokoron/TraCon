import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, tnorm_bool=False, snorm_bool=False, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                if self.tnorm_bool:
                    self.tn.append(TNorm(num_nodes, residual_channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(residual_channels))
                self.filter_convs.append(nn.Conv2d(in_channels=num * residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=num * residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            x_list = []
            x_list.append(x)
            b, c, n, t = x.shape
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            b, c, n, t = filter.shape
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x_ = F.relu(skip)
        rep = F.relu(self.end_conv_1(x_))
        out = self.end_conv_2(rep)
        return out, skip
