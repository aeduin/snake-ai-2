import torch
from torch import nn

import simulation
from simulation import World
from models import CnnAi

n_worlds = 256
episodes_count = 500
learning_rate = 0.001

device = torch.device('cuda:0')
world = World(10, 10, n_worlds, device)
model = CnnAi(world)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for episode_nr in range(episodes_count):
    print('start episode', episode_nr)
    world = World(10, 10, n_worlds, device)

    experience = []

    model.eval()

    total_reward = torch.tensor([0], device=device)

    n_steps = 0

    with torch.no_grad():
        while not torch.all(world.dead).cpu():
            alive = torch.logical_not(world.dead)
            network_input = world.space.to(torch.float)[alive]
            predicted_rewards = model(network_input)

            rewards_transpose = predicted_rewards.transpose(1, 0)
            
            # Don't take an action that results in certain death
            possible = torch.zeros((4, world.num_worlds), device=world.device, dtype=torch.bool)

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

                current_possible_large = torch.zeros((world.num_worlds,), device=world.device, dtype=torch.bool)
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
            impossible = torch.logical_not(possible[:, alive])
            impossible_large = torch.logical_not(possible)

            rewards_transpose[impossible] = -100

            actions_weight = torch.zeros((4, world.num_worlds), device=device)
            
            actions_weight[:, alive] = model.softmax(rewards_transpose)


            actions_weight += torch.randn(4, world.num_worlds, device=world.device) * model.temperature
            actions_weight[impossible_large] = -100

            taken_action_idx = torch.argmax(actions_weight, dim=0)

            dx = model.actions_x[taken_action_idx]
            dy = model.actions_y[taken_action_idx]


            reward = world.step(dx, dy)

            total_reward += torch.sum(reward)
            
            n_steps += 1

            if n_steps >= 10_000 and torch.sum(world.dead).cpu() <= n_worlds / 100 + 1:
                break

            if n_steps >= 20_000:
                break

            experience.append((network_input, alive, reward, taken_action_idx, torch.max(predicted_rewards, dim=1).values))

            # print(len(experience), ':', torch.sum(world.dead))

    print('n_steps =', n_steps)

    print('\tavg reward:', total_reward.cpu().numpy()[0] / world.num_worlds)

    model.train()

    total_loss = 0
    steps = 0

    goals = [torch.zeros(n_worlds, dtype=torch.float, device=device)]

    for _, _, has_reward, _, _ in experience:
        goals.append(goals[-1] + has_reward)

    print(torch.sum(goals[-1]) / n_worlds)

    for i in range(len(experience) - 1):
        network_input, alive, reward, taken_action_idx, _ = experience[i]
        # _, next_alive, _, _, max_predicted_next = experience[i + 1]

        # next_reward = torch.zeros(n_worlds, device=device)
        # next_reward[next_alive] = max_predicted_next
        # next_reward = next_reward[alive]
        # next_reward += reward[alive]

        goal = (goals[-(i + 1)][alive] > 0).to(torch.float)

        optimizer.zero_grad()

        # print(torch.arange(0, network_input.shape[0]), taken_action_idx)
        predicted_rewards = model(network_input)[torch.arange(0, network_input.shape[0]), taken_action_idx[alive]]

        loss = predicted_rewards - goal

        loss = torch.sum(loss * loss)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        steps += network_input.shape[0]

    print('loss =', total_loss / steps)       




