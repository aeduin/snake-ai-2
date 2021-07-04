import torch
import models
from simulation import World
import simulation
import numpy as np
from matplotlib import pyplot as plt

n_worlds = 1

device = torch.device('cpu')
world = World(7, 7, n_worlds, device)

world.snake_size[0] = 4
model = models.EquivariantAi(world)

world.step(torch.tensor([1], dtype=int), torch.tensor([0], dtype=int))
world.step(torch.tensor([1], dtype=int), torch.tensor([0], dtype=int))
world.step(torch.tensor([0], dtype=int), torch.tensor([1], dtype=int))
world.step(torch.tensor([0], dtype=int), torch.tensor([1], dtype=int))


alive = torch.logical_not(world.dead)
network_input = torch.zeros(n_worlds, simulation.num_channels + 1, world.width + 6, world.height + 6, device=device)[alive]
network_input[:, simulation.num_channels] = 1
network_input[:, -1, 3:-3, 3:-3] = 0
network_input[:, :-1, 3:-3, 3:-3] = world.space.to(torch.float)[alive]
predicted_rewards = model(network_input)

image = np.zeros((13, 13, 3))

for x in range(13):
    for y in range(13):
        # image[x, y, 0] = (x + y) / 26
        # image[x, y, 1] = (x) / 13
        # image[x, y, 2] = (y) / 13
        channels =  network_input[0, :, x, y]
        if channels[simulation.food_channel] > 0:
            image[x, y, 1] = 1
        elif channels[simulation.snake_head_channel] > 0:
            image[x, y, 2] = 1
        elif channels[simulation.snake_channel] > 0:
            image[x, y, :] = (channels[simulation.snake_channel].item() + 1) / (world.snake_size.item())
        elif channels[simulation.num_channels] > 0:
            image[x, y, 0] = 1



plt.imshow(image)
plt.show()


