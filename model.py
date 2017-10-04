import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(BaseModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.LOAD_MODEL_IF_EXIST = kwargs.pop("load_model_if_exist", False)
        self.SAVE_MODEL = kwargs.pop("save_model", False)
        self.SAVE_MODEL_NAME = kwargs.pop("save_model_name", None)
        if self.LOAD_MODEL_IF_EXIST and self.SAVE_MODEL_NAME is None:
            raise Exception("")
        if self.SAVE_MODEL and self.SAVE_MODEL_NAME is None:
            raise Exception("")

    def forward(self, x):
        pass


class DQNModel(BaseModel):

    def __init__(self, in_dim, out_dim, **kwargs):
        super(DQNModel, self).__init__(in_dim, out_dim, **kwargs)
        create_model = True
        if self.LOAD_MODEL_IF_EXIST:
            try:
                self.model = torch.load(self.SAVE_MODEL_NAME)
                print("Model loaded from {}".format(self.SAVE_MODEL_NAME))
                create_model = False
            except:
                create_model = True

        if create_model:
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

    def save(self):
        if self.SAVE_MODEL:
            torch.save(self.model, self.SAVE_MODEL_NAME)
            print("Saved model to {}".format(self.SAVE_MODEL_NAME))
