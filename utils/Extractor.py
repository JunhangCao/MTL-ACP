import torch
import torch.nn as nn


class SharedExtractor(nn.Module):
    def __init__(self, in_shapes, cell_nums, out_shapes):
        super(SharedExtractor, self).__init__()
        self.l1 = nn.Linear(in_shapes, cell_nums)
        self.r1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(cell_nums, out_shapes)
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        x = self.r2(x)
        return x