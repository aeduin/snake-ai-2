import simulation
from simulation import World
import torch
from torch import nn

import numpy as np

from typing import Tuple
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

class LinearAi(nn.Module):
    def __init__(self, world: World):
        super(LinearAi, self).__init__()

        self.temperature = 0.03
        device = world.device

        hidden_channels = 12

        self.world_size = (world.width + 6, world.height + 6)

        input_size = self.world_size[0] * self.world_size[1] * (simulation.num_channels + 1)

        hidden_size = 1_000
        output_size = 4

        self.dense1 = nn.Linear(input_size, hidden_size).to(device)
        self.dense2 = nn.Linear(hidden_size, hidden_size).to(device)
        self.dense3 = nn.Linear(hidden_size, hidden_size).to(device)
        self.dense4 = nn.Linear(hidden_size, output_size).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

    def forward(self, x):
        y = x
        
        y = y.view(y.shape[0], -1)

        y = self.dense1(y)
        y = self.relu(y)
        
        y = self.dense2(y)
        y = self.relu(y)
        
        y = self.dense3(y)
        y = self.relu(y)

        y = self.dense4(y)
        y = self.relu(y)

        return y

class CnnAi(nn.Module):
    def __init__(self, world: World):
        super(CnnAi, self).__init__()

        self.temperature = 0.03
        device = world.device
        

        hidden_channels = 12

        self.world_size = (world.width + 6, world.height + 6)
        size = torch.tensor(self.world_size)
        self.conv1 = nn.Conv2d(simulation.num_channels + 1, hidden_channels, (3, 3)).to(device)
        size = (size - 2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, (3, 3)).to(device)
        size = (size - 2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, (3, 3)).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * hidden_channels), 4).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

    def forward(self, x):
        y = x
        
        y = self.conv1(y)
        y = self.relu(y)
        
        y = self.conv2(y)
        y = self.relu(y)
        
        y = self.conv3(y)
        y = self.relu(y)

        y = y.view(y.shape[0], -1)
        
        y = self.dense(y)
        # y = self.softmax(y)
        # y = self.sigmoid(y)
        y = self.relu(y)

        return y

class EquivariantAi(nn.Module):
    def __init__(self, world: World):
        super(EquivariantAi, self).__init__()

        self.temperature = 0.03
        device = world.device
        

        hidden_channels = 4
        groups_count = 8

        self.world_size = (world.width + 6, world.height + 6)
        size = torch.tensor(self.world_size)
        self.conv1 = P4MConvZ2(in_channels=simulation.num_channels + 1, out_channels=hidden_channels, kernel_size=3, stride=1, padding=0).to(device)
        size = (size - 2)
        self.conv2 = P4MConvP4M(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=0).to(device)
        size = (size - 2)
        self.conv3 = P4MConvP4M(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=0).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * hidden_channels * groups_count), 4).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

    def forward(self, x):
        y = x
        # print(y.shape)
        
        y = self.conv1(y)
        y = self.relu(y)
        # print(y.shape)
        
        y = self.conv2(y)
        y = self.relu(y)
        # print(y.shape)
        
        y = self.conv3(y)
        y = self.relu(y)
        # print(y.shape)

        y = y.view(y.shape[0], -1)
        
        y = self.dense(y)
        # y = self.softmax(y)
        # y = self.sigmoid(y)
        y = self.relu(y)

        return y


class PerfectAi(nn.Module):
    def __init__(self, world: World):
        super(PerfectAi, self).__init__()

        self.temperature = 0.01
        device = world.device
        

        hidden_channels = 40
        groups_count = 8

        self.world_size = (world.width + 6, world.height + 6)
        size = torch.tensor(self.world_size)
        self.conv1 = P4MConvZ2(in_channels=simulation.num_channels + 1, out_channels=hidden_channels, kernel_size=5, stride=1, padding=0).to(device)
        size = (size - 4)
        self.conv2 = P4MConvP4M(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=5, stride=1, padding=0).to(device)
        size = (size - 4)
        self.conv3 = P4MConvP4M(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=0).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * hidden_channels * groups_count), 4).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

    def forward(self, x):
        y = x
        # print(y.shape)
        
        y = self.conv1(y)
        y = self.relu(y)
        # print(y.shape)
        
        y = self.conv2(y)
        y = self.relu(y)
        # print(y.shape)
        
        y = self.conv3(y)
        y = self.relu(y)
        # print(y.shape)

        y = y.view(y.shape[0], -1)
        
        y = self.dense(y)
        # y = self.softmax(y)
        # y = self.sigmoid(y)
        y = self.relu(y)

        return y
