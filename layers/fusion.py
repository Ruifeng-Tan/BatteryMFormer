import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, x_in_features, y_in_feautres, project_dim):
        super(GatedFusion, self).__init__()
        self.linear1 = nn.Linear(x_in_features, project_dim, bias=False)
        self.linear2 = nn.Linear(y_in_feautres, project_dim, bias=True)
    
    def forward(self, x, y):
        gate = F.sigmoid(self.linear1(x)+self.linear2(y))
        out = gate*x + (1-gate)*y
        return out