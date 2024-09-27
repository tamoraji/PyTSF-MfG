from metrics.ETTh_metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, save_model_ETT, load_model_ETT
from experiments.exp_basic import Exp_Basic
from data_process.long_dataloader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

from models.LightTS import LightTS
from ptflops import get_model_complexity_info

class Exp_long(Exp_Basic):
    def __init__(self, args):
        super(Exp_long, self).__init__(args)

    def _build_model(self):

        if self.args.features == 'S':
            in_dim = 1
        elif self.args.features == 'M':
            in_dim = self.args.enc_in
        else:
            print('Error!')


        model = LightTS(
            lookback=self.args.seq_len,
            lookahead=self.args.pred_len,
            hid_dim=self.args.hiddim,
            num_node=self.args.enc_in,
            dropout=self.args.dropout,
            chunk_size=self.args.chunk_size
        )

        print(model)
        macs, params = get_model_complexity_info(
            model, (self.args.seq_len, self.args.enc_in), as_strings=False, verbose=False
        )
        print('Mac: {}, Params: {}'.format(macs, params))
        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                valid_data, batch_x, batch_y)

            loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())


            total_loss.append(loss)
        total_loss = np.average(total_loss)

        preds = np.array(preds)
        trues = np.array(trues)
        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1,
                                          true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1,
                                          pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(
            pred_scales, true_scales)
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
            mse, mae, rmse, mape, mspe, corr))
        print('denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
            mses, maes, rmses, mapes, mspes, corrs))


        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        writer = SummaryWriter(
            'event/run_ETTh/{}'.format(self.args.model_name))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                loss = criterion(pred, true)

                train_loss.append(loss.item())

                if (i+1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch)*train_steps - i)
                    print(
                        '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(
                epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)
            writer.add_scalar('test_loss', test_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping with setting{} and epoch{}".format(setting, epoch+1))
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)

        save_model_ETT(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        true_scales = true_scales.reshape(-1,
                                          true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1,
                                          pred_scales.shape[-2], pred_scales.shape[-1])

        mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
        maes, mses, rmses, mapes, mspes, corrs = metric(
            pred_scales, true_scales)
        print('normed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
            mse, mae, rmse, mape, mspe, corr))
        print('TTTT denormed mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
            mses, maes, rmses, mapes, mspes, corrs))

        # result save
        if self.args.save:
            folder_path = 'exp/ett_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe, corr = metric(preds, trues)
            print('Test:mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}, corr:{:.4f}'.format(
                mse, mae, rmse, mape, mspe, corr))

            np.save(folder_path + 'metrics.npy',
                    np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
            np.save(folder_path + 'pred_scales.npy', pred_scales)
            np.save(folder_path + 'true_scales.npy', true_scales)

            np.savetxt(f'{folder_path}/pred.csv', preds[0], delimiter=",")
            np.savetxt(f'{folder_path}/true.csv', trues[0], delimiter=",")
            np.savetxt(f'{folder_path}/pred_scales.csv',
                       pred_scales[0], delimiter=",")
            np.savetxt(f'{folder_path}/true_scales.csv',
                       true_scales[0], delimiter=",")

        return mae, maes, mse, mses

    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.double().cuda()
        batch_y = batch_y.double()

        outputs = self.model(batch_x)

        # if self.args.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        return outputs, outputs_scaled, 0, 0, batch_y, batch_y_scaled
