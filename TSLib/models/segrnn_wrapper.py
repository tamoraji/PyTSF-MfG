from neuralforecast.models.base_model import PytorchForecastModel
from SegRNN import Model as SegRNNModel
import torch
import pandas as pd
import numpy as np


class SegRNNWrapper(PytorchForecastModel):
    def __init__(self, h, input_size, seg_len=5, d_model=512, dropout=0.1, futr_exog_list=None, stat_exog_list=None,
                 hist_exog_list=None):
        super().__init__(h, input_size, futr_exog_list, stat_exog_list, hist_exog_list)

        class Config:
            seq_len = input_size
            pred_len = h
            enc_in = 1  # Assuming univariate time series
            d_model = d_model
            dropout = dropout
            seg_len = seg_len
            task_name = 'long_term_forecast'

        self.model = SegRNNModel(Config())

    def forward(self, x):
        # Reshape input to match SegRNN expectations
        x = x.unsqueeze(-1)  # Add channel dimension
        return self.model.forecast(x).squeeze(-1)

    def fit(self, train_dataset, val_dataset=None, n_epochs=100, batch_size=32, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(n_epochs):
            self.model.train()
            for batch in train_dataset.batch(batch_size):
                optimizer.zero_grad()
                y_pred = self(batch['x'])
                loss = criterion(y_pred, batch['y'])
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            forecast = self(torch.from_numpy(X.astype(np.float32)))
        return forecast.numpy()


def SegRNN(h, input_size=100, seg_len=5, d_model=512, dropout=0.1):
    return SegRNNWrapper(h=h, input_size=input_size, seg_len=seg_len, d_model=d_model, dropout=dropout)