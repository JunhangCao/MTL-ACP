import torch.nn as nn


class TaskSpecificClassifier(nn.Module):
    def __init__(self, in_shapes, cell_nums, out_shapes):
        super(TaskSpecificClassifier, self).__init__()
        self.l1 = nn.Linear(in_shapes, cell_nums)
        self.s1 = nn.Sigmoid()
        self.l2 = nn.Linear(cell_nums, out_shapes)

    def forward(self,x):
        x = self.l1(x)
        x = self.s1(x)
        x = self.l2(x)
        return x