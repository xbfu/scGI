import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, input_dim, emb_size=32):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(input_dim, emb_size))
        nn.init.xavier_uniform_(self.emb.data)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        square_sum = (x @ self.emb) ** 2
        sum_square = ((x * x) @ (self.emb * self.emb))
        second_term = (square_sum - sum_square).sum(dim=1) / 2
        # x = self.l1(x).squeeze() + second_term
        x = second_term + self.bias
        return x
