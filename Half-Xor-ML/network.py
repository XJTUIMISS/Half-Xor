import torch
import pickle
import torch.optim as optim
import time
import numpy as np
import os

class Model(torch.nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h) #2 hidden layer
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        # x = self.relu3(x)
        # x = torch.log(x)
        return x


def testModel(model):
    test_n = 796712.0
    test_list = [0.485, 0.495, 0.499, 0.47, 0.499, 0.494, 0.479, 0.469, 0.335, 0.205, 0.105, 0.067, 0.032, 0.018, 0.01, 0.004, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    test_output = torch.exp(model(torch.tensor(test_list, dtype = torch.float32)))
    test_output = model(torch.tensor(test_list, dtype = torch.float32))

    print("actual: ", test_n, "  estimate: ", test_output.item(), "  error: ", abs(test_n - test_output.item()) / test_n)