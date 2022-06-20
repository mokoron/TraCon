import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.GTS.lib import utils
from models.GTS.model.pytorch.model import GTSModel
from models.GTS.model.pytorch.loss import metric, masked_mae_loss
import pandas as pd
import os
import time
from models import util as models_util
import util as u
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GTSSupervisor:
    def __init__(self, config, **kwargs):
        self.config = config
        self.opt = config.optimizer
        root = u.get_root()
        self.save_dir = f'{root}{config.save}/GTS/{config.dataset_name}'
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        self.expid = config.expid
        self.loss = masked_mae_loss

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        #self._writer = SummaryWriter('runs/' + self._log_dir)
        self._writer = SummaryWriter(self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(self.config.dataset_name, self.config.batch_size, self.config.batch_size)
        self.standard_scaler = self._data['scaler']

        ### Feas
        filepath = f"{root}ml_template/data/{config.dataset_name.lower()}.h5" # HANNOVER_DATA. METR-LA, PEMS-BAY
        df = pd.read_hdf(filepath)
        # Important for Hannover data that has null values
        if df.isnull().values.any():
            df = df.fillna(80)
            print('Warning: NaN Values in Data found. Filled them with 80!')

        num_samples = df.shape[0]
        num_train = round(num_samples * 0.7)
        df = df[:num_train].values
        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)

        k = self._train_kwargs.get('knn_k')
        knn_metric = 'cosine'
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        #self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))

        # setup model
        GTS_model = GTSModel(self.config, self._logger, **self._model_kwargs)
        self.GTS_model = GTS_model.cuda() if torch.cuda.is_available() else GTS_model
        self._logger.info("Model created")

        self._epoch_num = 0
        if self._epoch_num > 0:
            self.load_model()

    #@staticmethod
    def _get_log_dir(self, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            #horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'GTS_%s_%d_h_%d_%s_lr_%g_bs_%d_exp_%d/' % (
                filter_type_abbr, max_diffusion_step, self.config.output_horizon,
                structure, self.config.learning_rate, self.config.batch_size,
                self.config.expid)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self):
        # the best model is overwritten for the same expid
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        config = dict(self._kwargs)
        config['model_state_dict'] = self.GTS_model.state_dict()
        torch.save(config, f'{self.save_dir}/best%d.tar' % self.expid)
        self._logger.info("GTS: Saved model at {}".format(self.expid))
        return f'{self.save_dir}/best%d.tar' % self.expid

    def load_model(self):
        assert os.path.exists(f'{self.save_dir}/best%d.tar' % self.expid), 'Weights at best %d not found' % self.expid
        checkpoint = torch.load(f'{self.save_dir}/best%d.tar' % self.expid, map_location='cpu')
        self.GTS_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("GTS: Loaded model at {}".format(self.expid))

    def _setup_graph(self):
        with torch.no_grad():
            self.GTS_model = self.GTS_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y, _) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                self.GTS_model(x, self._train_feas)

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(base_lr=self.config.learning_rate, **kwargs)

    def evaluate(self, label, phase='val', batches_seen=0, gumbel_soft=True, ep=None):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """

        sample_rate = 5 if self.config.dataset_name.upper() in ["METR-LA", "PEMS-BAY"] else 15

        if phase == "val":
            self.GTS_model = self.GTS_model.eval()

            mean_loss, mean_mape, mean_rmse = self.get_metrics(label=label, temp=self.config.temperature, gumbel_soft=gumbel_soft, sample_rate=sample_rate, phase=phase, ep=ep)
        else:
            # test

            # load the best model in case of test
            self.load_model()

            mean_loss, mean_mape, mean_rmse = self.get_metrics(label=label, temp=self.config.temperature, gumbel_soft=gumbel_soft, sample_rate=sample_rate, phase=phase, ep=ep)

        self._writer.add_scalar('{} loss'.format(phase), mean_loss, batches_seen)
        return mean_loss, mean_mape, mean_rmse

    def get_metrics(self, label, temp, gumbel_soft, sample_rate, phase, ep=None):
        epoch = {"val": f"epoch_{ep}/", "test": ""}
        outputs = []
        realy = torch.Tensor(self._data[f'y_{phase}']).to(device)  # shape (num_samples, horizon, num_sensor, input_dim)
        realy = realy.transpose(1, 3)[:, 0, :, :]
        print("realy shape: ", realy.shape)  # shape (num_samples, num_sensor, horizon)

        for iter, (x, y, _) in enumerate(self._data[f'{phase}_loader'].get_iterator()):
            x, y = self._prepare_data(x, y)
            with torch.no_grad():
                preds, mid_output, embs = self.GTS_model(
                    label, x, self._train_feas, temp, gumbel_soft)  # output.shape (horizon, batch_size, num_sensor)
                preds = preds.permute(1, 2, 0)  # preds.shape (batch_size, num_sensor, horizon)
            outputs.append(preds)

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        al = []
        am = []
        ar = []
        for i in range(self.config.output_horizon):
            y_pred = self.standard_scaler.inverse_transform(yhat[:, :, i])
            y_true = realy[:, :, i]

            params = {"phase": phase, "model_name": self.config.model_name, "dataset_name": self.config.dataset_name,
                      "horizon": i, "epoch": epoch[phase], "save_loss": self.config.save_loss,
                      "save_pred_real": self.config.save_pred_real}
            metrics = metric(y_pred, y_true, params=params)

            if phase == "test":
                al.append(metrics[0])
            else:
                if label == 'without_regularization':
                    loss = self._compute_loss(y_true, y_pred)
                else:
                    loss_1 = self._compute_loss(y_true, y_pred)
                    pred = torch.sigmoid(mid_output.view(
                        mid_output.shape[0] * mid_output.shape[1]))
                    true_label = self.adj_mx.view(
                        mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                # save loss here
                if params is not None:
                    models_util.save_element(loss, params, "loss", "loss")
                al.append(loss.item())

            am.append(metrics[1])
            ar.append(metrics[2])

        mean_al = np.mean(al)
        mean_am = np.mean(am)
        mean_ar = np.mean(ar)

        if phase == "test":
            # Followed the DCRNN TensorFlow Implementation
            for i in range(self.config.output_horizon):
                message = 'GTS - dataset: {}. Horizon {}mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(self.config.dataset_name, (i + 1) * sample_rate,
                                                                                           al[i], am[i], ar[i])
                self._logger.info(message)

        return mean_al, mean_am, mean_ar

    def _train(self, base_lr, steps, lr_decay_ratio=0.1, epsilon=1e-8, **kwargs):
        his_loss = []
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('GTS: Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("GTS: num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(1, self.config.epochs+1):
            print("Num of epoch:",epoch_num)
            self.GTS_model = self.GTS_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            temp = self.config.temperature
            gumbel_soft = True

            if epoch_num < self.config.epoch_use_regularization:
                label = 'with_regularization'
            else:
                label = 'without_regularization'

            for batch_idx, (x, y, _) in enumerate(train_iterator):
                # print(x.shape, y.shape)
                # x.shape = [batch_size, input_horizon, num_nodes, input_dim], y.shape = [batch_size, input_horizon, num_nodes, input_dim]
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output, mid_output, _ = self.GTS_model(label, x, self._train_feas, temp, gumbel_soft, y, batches_seen)

                if (epoch_num % self.config.epochs) == self.config.epochs - 1:
                    output, mid_output, _ = self.GTS_model(label, x, self._train_feas, temp, gumbel_soft, y,
                                                            batches_seen)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.GTS_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.GTS_model.parameters(), lr=base_lr, eps=epsilon)

                self.GTS_model.to(device)

                y_pred = self.standard_scaler.inverse_transform(output)
                if label == 'without_regularization':
                    loss = self._compute_loss(y, y_pred)
                    losses.append(loss.item())
                else:
                    loss_1 = self._compute_loss(y, y_pred)
                    pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
                    true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)
                    loss = loss_1 + loss_g
                    losses.append((loss_1.item() + loss_g.item()))

                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.GTS_model.parameters(), self.max_grad_norm)

                optimizer.step()

            end_time = time.time()
            self._logger.info("GTS - dataset: {}. epoch complete, training time: {:.4f}/epoch".format(self.config.dataset_name, end_time - start_time))
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)
            lr_scheduler.step()

            self._logger.info("GTS: evaluating now!")
            # validation for this epoch
            start_time2 = time.time()
            val_loss, val_mape, val_rmse = self.evaluate(label, phase='val', batches_seen=batches_seen, gumbel_soft=gumbel_soft, ep=epoch_num)
            end_time2 = time.time()
            his_loss.append(val_loss)

            message = 'GTS - dataset: {}. Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                              'validation time: {:.1f}s'.format(self.config.dataset_name, epoch_num, self.config.epochs, batches_seen,
                                                        np.mean(losses), val_loss, val_mape, val_rmse,
                                                        lr_scheduler.get_lr()[0],
                                                        (end_time2 - start_time2))
            self._logger.info(message)

            # saving of the model
            if val_loss < min_val_loss:
                model_file_name = self.save_model()
                self._logger.info(
                        'GTS - dataset: {}. Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(self.config.dataset_name, min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                # remove all folders different from the best one
                models_util.clean_directory(config=self.config, index=epoch_num, phase='val')

        self._epoch_num = np.argmin(his_loss)
        models_util.clean_directory(config=self.config, index=self._epoch_num + 1, phase="val", rename=True)

        # testing
        test_loss, test_mape, test_rmse = self.evaluate(label, phase='test', batches_seen=batches_seen,
                                                            gumbel_soft=gumbel_soft)
        message = 'GTS - dataset: {}. Final result after {} epochs: train_mae: {:.4f}, val_mae: {:.4f}, test_mae: ' \
            '{:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, bestid: {}'.format(
            self.config.dataset_name, self.config.epochs, np.mean(losses), his_loss[self._epoch_num], test_loss,
            test_mape, test_rmse, lr_scheduler.get_lr()[0], self._epoch_num)
        self._logger.info(message)
        return [test_loss, test_mape, test_rmse], self._epoch_num

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.config.seq_length, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.config.output_horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_pred):
        loss = self.loss(y_pred, y_true)

        return loss
