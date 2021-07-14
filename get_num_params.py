import torch

import models
import simulation

def num_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    world = simulation.World(7, 7, 256, torch.device('cpu'))

    model = models.CnnAi(world)
    print('cnn:', num_parameters(model))

    model = models.LinearAi(world)
    print('linear:', num_parameters(model))

    model = models.EquivariantAi(world)
    print('equivariant:', num_parameters(model))

    model = models.LargeCnnAi(world)
    print('large cnn:', num_parameters(model))


