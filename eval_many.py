import torch

import simulation
from simulation import World
import models

import sys

random_action_probability = 0.03
random_weight_sd = 0.03

world_width = 7
world_height = 7
n_worlds = 2048

max_steps = 5_000

device = torch.device('cuda:0')
world = World(world_width, world_height, n_worlds, device)

file_name = sys.argv[1]
print(file_name)

model_name = ''
# model_name = 'lin'
for name in ['lin', 'eqv', 'cnn', 'lcn']:
    if name in file_name:
        model_name = name

if model_name == 'cnn':
    model = models.CnnAi(world)
elif model_name == 'eqv':
    model = models.EquivariantAi(world)
elif model_name == 'lin':
    model = models.LinearAi(world)
elif model_name == 'lcn':
    model = models.LargeCnnAi(world)
else:
    raise Exception('Invalid model name')

model.load_state_dict(torch.load(file_name))
model.eval()

total_reward = torch.zeros(n_worlds, device=device)

n_steps = 0

with torch.no_grad():
    while not torch.all(world.dead).item():
        alive = torch.logical_not(world.dead)
        network_input = torch.zeros(n_worlds, simulation.num_channels + 1, world.width + 6, world.height + 6, device=device)[alive]
        network_input[:, simulation.num_channels] = 1
        network_input[:, :-1, 3:-3, 3:-3] = world.space.to(torch.float)[alive]
        network_input[:, -1, 3:-3, 3:-3] = 0
        predicted_rewards = model(network_input)
        # predicted_rewards = torch.zeros(n_worlds, 4, device=device)[alive]

        rewards_transpose = predicted_rewards.transpose(1, 0)
        
        # Don't take an action that results in certain death
        possible = torch.zeros((4, world.num_worlds), device=device, dtype=torch.bool)

        for i in range(4):
            dx, dy = model.actions_cpu[i]
            new_head_x = world.snake_head_x[alive] + dx
            new_head_y = world.snake_head_y[alive] + dy
            
            current_possible = torch.logical_not(torch.logical_or(torch.logical_or(torch.logical_or(
                new_head_x < 0,
                new_head_x >= world.size[0]),
                new_head_y < 0),
                new_head_y >= world.size[1])
            )

            current_possible_large = torch.zeros((world.num_worlds,), device=device, dtype=torch.bool)
            current_possible_large[alive] = current_possible

            # alive_and_possible = alive[:]
            # alive_and_possible[alive] = current_possible

            current_possible_large[current_possible_large] = \
                world.space[
                    current_possible_large,
                    simulation.snake_channel,
                    new_head_x[current_possible],
                    new_head_y[current_possible]
                ] == 0
                # ] != world.snake_size[current_possible_large] - 1

            possible[i, :] = current_possible_large
        
        possible_transpose = possible[:, alive].transpose(1, 0)
        impossible = torch.logical_not(possible[:, alive]).transpose(1, 0)
        impossible_large = torch.logical_not(possible).transpose(1, 0)

        predicted_rewards[impossible] = -100

        actions_weight = torch.zeros((world.num_worlds, 4), device=device)
        
        # actions_weight[alive] = predicted_rewards
        actions_weight[alive] = torch.softmax(predicted_rewards, dim=1)
        actions_weight += torch.randn(world.num_worlds, 4, device=device) * random_weight_sd

        # actions_weight += torch.randn(4, world.num_worlds, device=device) * model.temperature
        randomize = torch.rand(world.num_worlds, device=device) < random_action_probability
        actions_weight[randomize] = torch.rand(world.num_worlds, 4, device=device)[randomize]
        actions_weight[impossible_large] = -100

        taken_action_idx = torch.argmax(actions_weight, dim=1)

        dx = model.actions_x[taken_action_idx]
        dy = model.actions_y[taken_action_idx]


        reward = world.step(dx, dy)

        total_reward += reward
        
        n_steps += 1

        # if n_steps >= 10_000 and torch.sum(world.dead).cpu() <= n_worlds / 100 + 1:
            # break

        if n_steps >= max_steps:
            print('worlds left:', world.num_worlds - torch.sum(world.dead).item())
            break

        # print(len(experience), ':', torch.sum(world.dead))

print('n_steps =', n_steps)

max_reward = torch.max(total_reward, 0).values.item()
avg_reward = torch.sum(total_reward).item() / world.num_worlds

print(f'{avg_reward=}')
print(f'{max_reward=}')

