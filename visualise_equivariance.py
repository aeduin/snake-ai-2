import torch
import models
from simulation import World
import simulation
import numpy as np
from matplotlib import pyplot as plt

torch.random.manual_seed(0)

def show_hidden(activations, group_id):
    image_hidden = activations[0, 0, group_id]
    hidden_max = torch.max(image_hidden)
    image_hidden /= hidden_max

    num_rotations = group_id % 4
    flip = group_id >= 4

    image_hidden = torch.rot90(image_hidden, k=4-num_rotations, dims=(0, 1))
    
    if flip:
        image_hidden = torch.flip(image_hidden, dims=[0])


    image_hidden = image_hidden.detach().numpy()
    result = np.zeros(image_hidden.shape + (3,))

    for i in range(3):
        result[:, :, i] = image_hidden

    plt.imshow(result)
    plt.show()



n_worlds = 1

device = torch.device('cpu')
world = World(7, 7, n_worlds, device)

world.snake_size[0] = 4
model = models.EquivariantAi(world)
model.load_state_dict(torch.load('./model_output/best_model_eqv_2021-07-05_07:19:34_983'))

world.step(torch.tensor([1], dtype=int), torch.tensor([0], dtype=int))
world.step(torch.tensor([1], dtype=int), torch.tensor([0], dtype=int))
world.step(torch.tensor([0], dtype=int), torch.tensor([1], dtype=int))
world.step(torch.tensor([0], dtype=int), torch.tensor([1], dtype=int))


alive = torch.logical_not(world.dead)
network_input = torch.zeros(n_worlds, simulation.num_channels + 1, world.width + 6, world.height + 6, device=device)[alive]
network_input[:, simulation.num_channels] = 1
network_input[:, -1, 3:-3, 3:-3] = 0
network_input[:, :-1, 3:-3, 3:-3] = world.space.to(torch.float)[alive]

image_input = np.zeros((13, 13, 3))


for x in range(13):
    for y in range(13):
        # image[x, y, 0] = (x + y) / 26
        # image[x, y, 1] = (x) / 13
        # image[x, y, 2] = (y) / 13
        channels =  network_input[0, :, x, y]
        if channels[simulation.food_channel] > 0:
            image_input[x, y, 1] = 1
        elif channels[simulation.snake_head_channel] > 0:
            image_input[x, y, 2] = 1
        elif channels[simulation.snake_channel] > 0:
            image_input[x, y, :] = (channels[simulation.snake_channel].item() + 1) / (world.snake_size.item())
        elif channels[simulation.num_channels] > 0:
            image_input[x, y, 0] = 1

plt.imshow(image_input)
plt.show()

image_hidden = model.relu(model.conv1(network_input))

for i in range(8):
    show_hidden(image_hidden, i)
'''
image_hidden = model.relu(model.conv2(image_hidden))
for i in range(8):
    show_hidden(image_hidden, i)
'''
