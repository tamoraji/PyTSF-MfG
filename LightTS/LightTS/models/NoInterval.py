import torch
import torch.nn as nn
import torch.nn.functional as F


class NoIntervalLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node, c_dim=None):
        super(NoIntervalLayer, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        if c_dim is None:
            self.c_dim = self.num_node // 2
        else:
            self.c_dim = c_dim

        self._build()

    def _build(self):
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )

        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)

        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)


    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))

        x = x.permute(0, 2, 1)

        return x


class NoInterval(nn.Module):
    def __init__(self, lookback, lookahead, hid_dim, num_node, dropout=0, chunk_size=40, c_dim=40):
        super(NoInterval, self).__init__()

        self.lookback = int(lookback)
        self.lookahead = int(lookahead)

        self.chunk_size = chunk_size
        assert(lookback % chunk_size == 0)
        self.num_chunks = lookback // chunk_size

        self.hid_dim = int(hid_dim)
        self.num_node = int(num_node)
        self.c_dim = int(c_dim)
        self.dropout = dropout
        self._build()

    def _build(self):
        self.layer_1 = NoIntervalLayer(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim // 4,
            output_dim=self.hid_dim // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = NoIntervalLayer(
            input_dim=self.chunk_size,
            hid_dim=self.hid_dim // 4,
            output_dim=self.hid_dim // 4,
            num_node=self.num_chunks
        )

        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = NoIntervalLayer(
            input_dim=self.hid_dim // 4,
            hid_dim=self.hid_dim // 4,
            output_dim=self.lookahead,
            num_node=self.num_node,
            c_dim=self.c_dim
        )

        # self.ar = nn.Sequential(
        #     nn.Linear(self.lookback, self.hid_dim //4),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hid_dim // 4, self.lookahead)
        # )
        self.ar = nn.Linear(self.lookback, self.lookahead)

    def forward(self, x):
        B, T, N = x.size()

        highway = self.ar(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)

        # continuous sampling
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)
        
        # interval sampling
        # x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        # x2 = x2.permute(0, 3, 1, 2)
        # x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        # x2 = self.layer_2(x2)
        # x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = x1

        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)

        out = self.layer_3(x3)

        out = out + highway
        return out
