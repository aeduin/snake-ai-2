import simulation
import torch
import models

device = torch.device('cpu')
w = simulation.World(7, 7, 1, device)
n_worlds = 1

model = models.EquivariantAi(w)

w.step(torch.tensor([0]), torch.tensor([1]))
w.step(torch.tensor([0]), torch.tensor([1]))
w.step(torch.tensor([1]), torch.tensor([0]))

alive = torch.logical_not(w.dead)
network_input = torch.zeros(n_worlds, simulation.num_channels + 1, w.width + 6, w.height + 6, device=device)[alive]
network_input[:, simulation.num_channels] = 1
network_input[:, :-1, 3:-3, 3:-3] = w.space.to(torch.float)[alive]

for rotations in range(4):
    prediction = model(torch.rot90(network_input, rotations, (2, 3)))
    print(prediction)
