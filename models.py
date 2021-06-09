import simulation
from simulation import World
import torch
from torch import nn

import numpy as np

from typing import Tuple

class CenteredRotated(nn.Module):
    def __init__(self, world_size, num_worlds, device):
        super(CenteredRotated, self).__init__()
        

        self.world_size = (int(world_size[0]), int(world_size[1]))
        size = world_size * 2 - 1
        self.conv1 = nn.Conv2d(simulation.num_channels, 8, (5, 5)).to(device)
        size = (size - 4) // 2
        self.conv2 = nn.Conv2d(8, 12, (5, 5)).to(device)
        size = (size - 4)
        self.conv3 = nn.Conv2d(12, 16, (3, 3)).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * 16), 1).to(device)

        self.pool = nn.MaxPool2d((2, 2))

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

    def evaluate(self, x):
        y = x

        y = self.conv1(y)
        y = self.pool(y)
        
        y = self.conv2(y)
        
        y = self.conv3(y)

        y = y.view(y.shape[0], -1)
        
        y = self.dense(y)

        return y

    def forward(self, world: World):
        network_input = torch.zeros((world.num_worlds, simulation.num_channels, self.world_size[0] * 2 - 1, self.world_size[1] * 2 - 1), dtype=torch.int16, device=world.device)
        
        # Decide the range of where to put the data
        from_x = (self.world_size[0] - world.snake_head_x - 1).to(torch.long)
        from_y = (self.world_size[1] - world.snake_head_y - 1).to(torch.long)

        to_x = (from_x + world.size[0]).to(torch.long)
        to_y = (from_y + world.size[1]).to(torch.long)
        
        # print(from_x)
        # print(from_y)
        # print(to_x)
        # print(to_y)

        # print(network_input[4:14, 4:14])

        network_input[0, from_x:to_x, from_y:to_y] = world.space
        network_input = network_input.to(torch.float)

        # network_input = network_input.permute(0, 3, 1, 2)

        result = torch.zeros((4,), dtype=torch.float, device=world.device)
        
        result[0] = self.evaluate(network_input)
        for i in range(1, 4):
            network_input = torch.rot90(network_input, dims=(2, 3))
            result[i] = self.evaluate(network_input)
 
        # print(result)

        possible = torch.zeros((4,), device=world.device, dtype=torch.bool)
        for i in range(4):
            dx, dy = self.actions_cpu[i]
            new_head_x = world.snake_head_x + dx
            new_head_y = world.snake_head_y + dy
            possible[i] = not (
                new_head_x < 0 or new_head_x >= world.size[0] or new_head_y < 0 or new_head_y >= world.size[1] or world.space[new_head_x, new_head_y, simulation.snake_channel] > 0
            )
        
        possible_actions = self.actions[possible]
        possible_result = result[possible]

        return possible_actions[torch.argmax(possible_result)]

class RotatedAI(nn.Module):
    def __init__(self, world_size, device):
        super(RotatedAI, self).__init__()

        self.temperature = 0.05
        

        self.world_size = (int(world_size[0]), int(world_size[1]))
        size = world_size
        self.conv1 = nn.Conv2d(simulation.num_channels, 8, (5, 5)).to(device)
        size = (size - 4)
        self.conv2 = nn.Conv2d(8, 12, (3, 3)).to(device)
        size = (size - 2)
        self.conv3 = nn.Conv2d(12, 16, (3, 3)).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * 16), 1).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

        self.old_network_input = None

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

        return y

    def world_to_input(self, world: World):
        return world.space.to(torch.float)

    def get_actions(self, world: World) -> Tuple[torch.Tensor, torch.Tensor]:
        network_input = self.world_to_input(world)

        result = torch.zeros((4, world.num_worlds), dtype=torch.float, device=world.device)

        result[0] = self(network_input).view((-1,))
        for i in range(1, 4):
            network_input = torch.rot90(network_input, dims=(2, 3))
            result[i] = self(network_input).view((-1,))

        possible = torch.zeros((4, world.num_worlds), device=world.device, dtype=torch.bool)
        for i in range(4):
            dx, dy = self.actions_cpu[i]
            new_head_x = world.snake_head_x + dx
            new_head_y = world.snake_head_y + dy
            current_possible = torch.logical_not(torch.logical_or(torch.logical_or(torch.logical_or(
                new_head_x < 0,
                new_head_x >= world.size[0]),
                new_head_y < 0),
                new_head_y >= world.size[1])
            )

            current_possible[current_possible] = \
                world.space[
                    current_possible,
                    simulation.snake_channel,
                    new_head_x[current_possible],
                    new_head_y[current_possible]
                ] == 0

            possible[i] = current_possible
        
        # possible_actions = self.actions[possible]
        # possible_result = result[possible]
        actions_weight = self.softmax(result)

        actions_weight += torch.rand(4, world.num_worlds, device=world.device) * self.temperature

        actions_weight[torch.logical_not(possible)] = -100

        taken_action_idx = torch.argmax(actions_weight, dim=0)
        
        return self.actions_x[taken_action_idx], self.actions_y[taken_action_idx]

class CnnAi(nn.Module):
    def __init__(self, world: World):
        super(CnnAi, self).__init__()

        self.temperature = 0.1
        device = world.device
        

        self.world_size = (world.width, world.height)
        size = torch.tensor(self.world_size)
        self.conv1 = nn.Conv2d(simulation.num_channels, 8, (5, 5)).to(device)
        size = (size - 4)
        self.conv2 = nn.Conv2d(8, 12, (3, 3)).to(device)
        size = (size - 2)
        self.conv3 = nn.Conv2d(12, 16, (3, 3)).to(device)
        size = (size - 2)
        self.dense = nn.Linear(int(size[0] * size[1] * 16), 4).to(device)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

         
        self.actions_cpu = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        self.actions = torch.Tensor(self.actions_cpu).to(device, torch.long)

        self.actions_x = torch.tensor([x for x, _ in self.actions_cpu], device=device, dtype=torch.long)
        self.actions_y = torch.tensor([y for _, y in self.actions_cpu], device=device, dtype=torch.long)

        self.old_network_input = None

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

        return y




if __name__ == "__main__":
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    w = World(10, 10, 4, device)

    model = RotatedAI(w.size, device)
    model.eval()
    
    with torch.no_grad():
        print("start eval")
        for i in range(20):
            action_x, action_y = model.get_actions(w)
            w.step(action_x, action_y)

            # print(w)
            # input()

            if i == 0:
                print("done first eval")
            print(w)
            input()
