import simulation
import torch
import models

device = torch.device('cpu')
w = simulation.World(7, 7, 1, device)
n_worlds = 1

model = models.EquivariantAi(w)
model.load_state_dict(torch.load('./model_output/best_model_eqv_2021-07-03_17:17:42_18'))

w.step(torch.tensor([0]), torch.tensor([1]))
w.step(torch.tensor([0]), torch.tensor([1]))
w.step(torch.tensor([1]), torch.tensor([0]))

alive = torch.logical_not(w.dead)
network_input = torch.zeros(n_worlds, simulation.num_channels + 1, w.width + 6, w.height + 6, device=device)[alive]
network_input[:, simulation.num_channels] = 1
network_input[:, :-1, 3:-3, 3:-3] = w.space.to(torch.float)[alive]

print(w)
w.space = torch.flip(w.space, dims=[2])
print(w)

for rotations in range(4):
    prediction = model(torch.rot90(network_input, rotations, (2, 3)))
    prediction2 = model(torch.flip(torch.rot90(network_input, rotations, (2, 3)), dims=[2]))
    print(f'{rotations}, noflip=', prediction)
    print(f'{rotations}, flip=', prediction2)

    print(model.actions_cpu[torch.argmax(prediction).item()])
    print(model.actions_cpu[torch.argmax(prediction2).item()])
