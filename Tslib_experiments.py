import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from modules.utils import load_datasets_statforecast_uni
from modules.evaluator import Evaluator
from modules.results_saver import ResultsSaver
from modules.performance_utils import measure_time_and_memory
from modules.algorithm_factory import create_algorithm
from modules.config import ALGORITHM_POOL, DATASET_POOL
from modules.plot_utils import plot_forecast_vs_actual
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set parameters
DATA_PATH = '/Users/moji/PyTSF-MfG/data'
OUTPUT_DIR = '/Users/moji/PyTSF-MfG/results'
PLOT_DIR = '/Users/moji/PyTSF-MfG/plots'

os.makedirs(PLOT_DIR, exist_ok=True)


def preprocess_data(data, seq_len, pred_len):
    logger.info(f"Preprocessing data. Input shape: {data.shape}")

    # Ensure data is a numpy array
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, pd.DataFrame):
        data = data.values

    logger.info(f"Data shape after conversion to numpy: {data.shape}")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    logger.info(f"Normalized data shape: {data_normalized.shape}")

    # Create input/output sequences
    X, y = [], []
    for i in range(len(data_normalized) - seq_len - pred_len + 1):
        X.append(data_normalized[i:(i + seq_len)])
        y.append(data_normalized[(i + seq_len):(i + seq_len + pred_len)])

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")

    return X, y, scaler


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f'Updating learning rate to {lr}')


def train_segrnn(model, train_loader, vali_loader, criterion, optimizer, args):
    logger.info("Starting SegRNN training...")

    time_now = time.time()
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            outputs = model(batch_x, None, None, None)  # Forward pass

            f_dim = -1 if args.features == 'MS' else 0
            loss = criterion(outputs[:, -args.pred_len:, f_dim:], batch_y[:, -args.pred_len:, f_dim:])
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            loss.backward()
            optimizer.step()

        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = validate(model, vali_loader, criterion, args)

        logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, args.checkpoint_path)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    return model


def validate(model, vali_loader, criterion, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            outputs = model(batch_x, None, None, None)

            f_dim = -1 if args.features == 'MS' else 0
            loss = criterion(outputs[:, -args.pred_len:, f_dim:], batch_y[:, -args.pred_len:, f_dim:])
            total_loss.append(loss.item())
    return np.average(total_loss)


def predict(model, test_loader, args):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            outputs = model(batch_x, None, None, None)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


def run_experiment(data, name, args):
    logger.info(f"\nStarting experiment for dataset: {name}")
    logger.info(f"Algorithm: {args.model}, Horizon: {args.pred_len}")
    logger.info(f"Initial data shape: {data.shape}")

    # Split the data
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Ensure test set length is a multiple of horizon
    test_length = (len(test_data) // args.pred_len) * args.pred_len
    test_data = test_data[:test_length]

    logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Preprocess data
    X_train, y_train, scaler = preprocess_data(train_data['y'], args.seq_len, args.pred_len)
    X_test, y_test, _ = preprocess_data(test_data['y'], args.seq_len, args.pred_len)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    vali_dataset = TensorDataset(torch.FloatTensor(X_test[:int(len(X_test) * 0.2)]),
                                 torch.FloatTensor(y_test[:int(len(y_test) * 0.2)]))
    vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(X_test[int(len(X_test) * 0.2):]),
                                 torch.FloatTensor(y_test[int(len(y_test) * 0.2):]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create the model
    model = create_algorithm(args.model, vars(args))
    logger.info(f"Model created: {type(model).__name__}")

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    train_start = time.time()
    model = train_segrnn(model, train_loader, vali_loader, criterion, optimizer, args)
    train_time = time.time() - train_start

    # Make predictions
    test_start = time.time()
    predictions, actuals = predict(model, test_loader, args)
    test_time = time.time() - test_start

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions.reshape(-1, args.pred_len)).reshape(-1)
    actuals = scaler.inverse_transform(actuals.reshape(-1, args.pred_len)).reshape(-1)

    logger.info(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")

    # Calculate metrics
    metrics = Evaluator.calculate_metrics(actuals, predictions)

    # Add computational complexity metrics
    metrics['train_time'] = train_time
    metrics['test_time'] = test_time

    # Generate plots
    plot_forecast_vs_actual(actuals, predictions, name, args.model, args.pred_len, PLOT_DIR)

    logger.info("Experiment completed")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegRNN Time Series Forecasting')
    parser.add_argument('--model', type=str, default='SegRNN', help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--enc_in', type=int, default=1, help='Encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model')
    parser.add_argument('--seg_len', type=int, default=5, help='Segment length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='Train epochs')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='Learning rate adjust type')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/',
                        help='Path for saving model checkpoints')
    parser.add_argument('--features', type=str, default='S',
                        help='Forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    args = parser.parse_args()

    # Load dataset
    datasets = load_datasets_statforecast_uni(DATA_PATH)
    if args.dataset not in datasets:
        raise ValueError(f"Dataset {args.dataset} not found")
    data = datasets[args.dataset]

    # Run experiment
    metrics = run_experiment(data, args.dataset, args)

    # Save results
    saver = ResultsSaver(OUTPUT_DIR)
    saver.save_results({f'horizon_{args.pred_len}': metrics}, args.model, args.pred_len, args.dataset)

    # Print summary of results
    print(f"Summary of results for {args.dataset}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    print("\nExperiment completed. Results saved")