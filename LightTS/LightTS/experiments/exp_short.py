from utils.math_utils import smooth_l1_loss
from metrics.ETTh_metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, save_model_financial, load_model_financial, save_results
from data_process.short_dataloader import DataLoaderH
from experiments.exp_basic import Exp_Basic
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import os
import math
import time
import warnings
warnings.filterwarnings('ignore')

from models.LightTS import LightTS
from models.NoChannel import NoChannel
from models.NoContinuous import NoContinuous
from models.NoInterval import NoInterval
from ptflops import get_model_complexity_info


class Exp_short(Exp_Basic):
    def __init__(self, args):
        super(Exp_short, self).__init__(args)
        if self.args.L1Loss:
            self.criterion = smooth_l1_loss
        else:
            self.criterion = nn.MSELoss(size_average=False).cuda()
        self.evaluateL2 = nn.MSELoss(size_average=False).cuda()
        self.evaluateL1 = nn.L1Loss(size_average=False).cuda()
        self.writer = SummaryWriter(
            '.exp/run_financial/{}'.format(args.model_name))

    def _build_model(self):
        if self.args.dataset_name == 'electricity':
            in_dim = 321

        if self.args.dataset_name == 'solar_AL':
            in_dim = 137

        if self.args.dataset_name == 'exchange_rate':
            in_dim = 8

        if self.args.dataset_name == 'traffic':
            in_dim = 862

        print(self.args.model_name)

        if self.args.model_name == 'LightTS':
            model = LightTS(
                lookback=self.args.window_size,
                lookahead=self.args.horizon,
                hid_dim=self.args.hiddim,
                num_node=in_dim,
                dropout=self.args.dropout,
                chunk_size=self.args.chunk_size,
                c_dim=self.args.c_dim
            )
        elif self.args.model_name == 'NoChannel':
            model = NoChannel(
                lookback=self.args.window_size,
                lookahead=self.args.horizon,
                hid_dim=self.args.hiddim,
                num_node=in_dim,
                dropout=self.args.dropout,
                chunk_size=self.args.chunk_size
            )
        elif self.args.model_name == 'NoInterval':
            model = NoInterval(
                lookback=self.args.window_size,
                lookahead=self.args.horizon,
                hid_dim=self.args.hiddim,
                num_node=in_dim,
                dropout=self.args.dropout,
                chunk_size=self.args.chunk_size
            )
        elif self.args.model_name == 'NoContinuous':
            model = NoContinuous(
                lookback=self.args.window_size,
                lookahead=self.args.horizon,
                hid_dim=self.args.hiddim,
                num_node=in_dim,
                dropout=self.args.dropout,
                chunk_size=self.args.chunk_size
            )

        macs, params = get_model_complexity_info(
            model, (self.args.window_size, in_dim), as_strings=False, verbose=False
        )
        print('Mac: {}, Params: {}'.format(macs, params))
        return model

    def _get_data(self, DATADIR):
        if self.args.dataset_name == 'electricity':
            self.args.data = f'{DATADIR}/datasets/short/electricity.txt'

        if self.args.dataset_name == 'solar_AL':
            self.args.data = f'{DATADIR}/datasets/short/solar_AL.txt'

        if self.args.dataset_name == 'exchange_rate':
            self.args.data = f'{DATADIR}/datasets/short/exchange_rate.txt'

        if self.args.dataset_name == 'traffic':
            self.args.data = f'{DATADIR}/datasets/short/traffic.txt'
        return DataLoaderH(self.args.data, 0.6, 0.2, self.args.horizon, self.args.window_size, self.args.normalize)

    def _select_optimizer(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    def train(self, DATADIR):

        best_val = 10000000

        optim = self._select_optimizer()

        data = self._get_data(DATADIR)
        X = data.train[0]
        Y = data.train[1]
        save_path = os.path.join(self.args.save_path, self.args.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        epoch_start = 0

        for epoch in range(epoch_start, self.args.epochs):
            epoch_start_time = time.time()
            iter = 0
            self.model.train()
            total_loss = 0
            n_samples = 0
            final_loss = 0
            # lr = adjust_learning_rate(optim, epoch, self.args)

            for tx, ty in data.get_batches(X, Y, self.args.batch_size, True):
                self.model.zero_grad()  # torch.Size([32, 168, 137])
                forecast = self.model(tx)
                scale = data.scale.expand(
                    forecast.size(0), self.args.horizon, data.m)
                bias = data.bias.expand(
                    forecast.size(0), self.args.horizon, data.m)
                # used with multi-step
                weight = torch.tensor(self.args.lastWeight).cuda()

                if self.args.single_step:  # single step
                    ty_last = ty[:, -1, :]
                    scale_last = data.scale.expand(forecast.size(0), data.m)
                    bias_last = data.bias.expand(forecast.size(0), data.m)
                    loss = self.criterion(
                        forecast[:, -1] * scale_last + bias_last, ty_last * scale_last + bias_last)

                else:

                    loss = self.criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                            ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                        + weight * self.criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                  ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])

                loss.backward()
                total_loss += loss.item()

                final_loss += loss.item()
                n_samples += (forecast.size(0) * data.m)
                grad_norm = optim.step()

                if iter % 100 == 0:
                    print('iter:{:3d} | loss: {:.7f}'.format(
                        iter, loss.item()/(forecast.size(0) * data.m)))
                    if iter % 500 == 0 and iter != 0 and self.args.dataset_name in []:
                        val_loss, val_rae, val_corr = self.validate(
                            data, data.valid[0], data.valid[1])
                        test_loss, test_rae, test_corr = self.validate(
                            data, data.test[0], data.test[1])
                        print(
                            '| EncoDeco: end of epoch {:3d} iter {:4d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                            ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                                epoch, iter, (time.time() - epoch_start_time), total_loss / n_samples, val_loss, val_rae,
                                val_corr, test_loss, test_rae, test_corr), flush=True)

                        if val_loss < best_val:
                            save_model_financial(epoch, self.args.lr, self.model, save_path,
                                       model_name=self.args.dataset_name, args=self.args)
                            print('--------------| Best Val loss |--------------')
                            best_val = val_loss
                iter += 1

            val_loss, val_rae, val_corr = self.validate(
                data, data.valid[0], data.valid[1])
            test_loss, test_rae, test_corr = self.validate(
                data, data.test[0], data.test[1])

            self.writer.add_scalar(
                'Train_loss_tatal', total_loss / n_samples, global_step=epoch)
            self.writer.add_scalar(
                'Train_loss_Final', final_loss / n_samples, global_step=epoch)
            self.writer.add_scalar(
                'Validation_final_rse', val_loss, global_step=epoch)
            self.writer.add_scalar(
                'Validation_final_rae', val_rae, global_step=epoch)
            self.writer.add_scalar(
                'Validation_final_corr', val_corr, global_step=epoch)
            self.writer.add_scalar(
                'Test_final_rse', test_loss, global_step=epoch)
            self.writer.add_scalar(
                'Test_final_rae', test_rae, global_step=epoch)
            self.writer.add_scalar(
                'Test_final_corr', test_corr, global_step=epoch)


            print(
                '| EncoDeco: end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), total_loss / n_samples, val_loss, val_rae, val_corr, test_loss, test_rae, test_corr), flush=True)

            if val_loss < best_val:
                save_model_financial(epoch, self.args.lr, self.model, save_path, model_name=self.args.dataset_name, args=self.args)
                print('--------------| Best Val loss |--------------')
                best_val = val_loss
        return total_loss / n_samples

    def validate(self, data, X, Y, evaluate=False):
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0

        total_loss_mid = 0
        total_loss_l1_mid = 0
        n_samples = 0
        predict = None
        res_mid = None
        test = None

        forecast_set = []
        Mid_set = []
        target_set = []

        if evaluate:
            save_path = os.path.join(self.args.save_path, self.args.model_name)
            self.model = load_model_financial(self.model, save_path, model_name=self.args.dataset_name, args=self.args)[0]

        for X, Y in data.get_batches(X, Y, self.args.batch_size, False):
            with torch.no_grad():
                forecast = self.model(X)
            # only predict the last step
            true = Y[:, -1, :].squeeze()
            output = forecast[:, -1, :].squeeze()

            forecast_set.append(forecast)
            target_set.append(Y)

            if len(forecast.shape) == 1:
                forecast = forecast.unsqueeze(dim=0)

            if predict is None:
                predict = forecast[:, -1, :].squeeze()
                test = Y[:, -1, :].squeeze()  # torch.Size([32, 3, 137])

            else:
                predict = torch.cat((predict, forecast[:, -1, :].squeeze(1)))
                test = torch.cat((test, Y[:, -1, :].squeeze(1)))

            scale = data.scale.expand(output.size(0), data.m)
            bias = data.bias.expand(output.size(0), data.m)

            total_loss += self.evaluateL2(output *
                                          scale + bias, true * scale + bias).item()
            total_loss_l1 += self.evaluateL1(output *
                                             scale + bias, true * scale + bias).item()

            n_samples += (output.size(0) * data.m)

        forecast_Norm = torch.cat(forecast_set, axis=0)
        target_Norm = torch.cat(target_set, axis=0)


        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae

        # only calculate the last step for financial datasets.
        predict = forecast_Norm.cpu().numpy()[:, -1, :]
        Ytest = target_Norm.cpu().numpy()[:, -1, :]

        if evaluate:
            p1 = f'{self.args.model_name}_{self.args.dataset_name}_la{self.args.horizon}_lb{self.args.window_size}_cs{self.args.chunk_size}_bs{self.args.batch_size}_lr{self.args.lr}_hid{self.args.hiddim}'
            p2 = f'{self.args.model_name}_{self.args.dataset_name}_la{self.args.horizon}'
            save_results(forecast_Norm, target_Norm, p1, p2, self.args.output_path)

        sigma_p = predict.std(axis=0)
        sigma_g = Ytest.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_p * sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)
                       ).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()

        print(
            '|valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(
                rse, rae, correlation), flush=True)

        return rse, rae, correlation


