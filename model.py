import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaseModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        pass


class DQNModel(BaseModel):

    def __init__(self, in_dim, out_dim):
        super(DQNModel, self).__init__(in_dim, out_dim)
        self.model = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        )

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)
