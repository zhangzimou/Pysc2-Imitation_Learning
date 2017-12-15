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


class DQNModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(DQNModel, self).__init__()
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


class AtariNet(nn.Module):

    def __init__(self, dim):
        super(AtariNet, self).__init__()
        if dim == 64:
            self.conv1 = nn.Conv2d(1, 16, 8, 4)
            self.conv2 = nn.Conv2d(16, 32, 4, 2)
            self.fc1 = nn.Linear(32*6*6, 256)
            self.fc2 = nn.Linear(256, 128)
        elif dim == 32:
            self.conv1 = nn.Conv2d(1, 4, 4, 2)
            self.conv2 = nn.Conv2d(4, 8, 4, 2)
            self.fc1 = nn.Linear(8*6*6, 64)
            self.fcx = nn.Linear(64, 32)
            self.fcy = nn.Linear(64, 32)
        else:
            raise Exception("")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        xx = self.fcx(x)
        xy = self.fcy(x)
        return torch.cat([xx, xy], 1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class SimpleNet(nn.Module):

    def __init__(self, dim):
        super(SimpleNet, self).__init__()
        self.fc_x = nn.Linear(2*dim, dim)
        self.fc_y = nn.Linear(2*dim, dim)

    def forward(self, x):
        xx = self.fc_x(x)
        xy = self.fc_y(x)
        return torch.cat([xx, xy], 1)


class MoveToBeaconTest(nn.Module):

    def __init__(self, dim):
        super(MoveToBeaconTest, self).__init__()
        self.dim = dim
        # self.fc = nn.Linear(2*dim, 2*dim)
        self.fc1 = nn.Linear(2*dim, 2*dim)
        self.fc_x = nn.Linear(2*dim, dim)
        self.fc_y = nn.Linear(2*dim, dim)
        # self.fc_x = nn.Linear(dim, dim)
        # self.fc_y = nn.Linear(dim, dim)

    def forward(self, x):
        # return torch.cat([self.fc_x(x[:, :self.dim]), self.fc_y(x[:, self.dim:])], 1)
        # return self.fc(x)
        x = F.relu(self.fc1(x))
        return torch.cat([self.fc_x(x), self.fc_y(x)], 1)


class ModelWrapper():

    def __init__(self, model, in_dim, out_dim, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.LOAD_MODEL_IF_EXIST = kwargs.pop("load_model_if_exist", False)
        self.SAVE_MODEL = kwargs.pop("save_model", False)
        self.SAVE_MODEL_NAME = kwargs.pop("save_model_name", None)
        if self.LOAD_MODEL_IF_EXIST and self.SAVE_MODEL_NAME is None:
            raise Exception("")
        if self.SAVE_MODEL and self.SAVE_MODEL_NAME is None:
            raise Exception("")

        create_model = True
        if self.LOAD_MODEL_IF_EXIST:
            try:
                self.model = torch.load(self.SAVE_MODEL_NAME)
                print("Model loaded from {}".format(self.SAVE_MODEL_NAME))
                create_model = False
            except:
                create_model = True

        if create_model:
            self.model = model

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, *args):
        return self.model.load_state_dict(*args)

    def save(self):
        if self.SAVE_MODEL:
            torch.save(self.model, self.SAVE_MODEL_NAME)
            print("Saved model to {}".format(self.SAVE_MODEL_NAME))
