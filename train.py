import torch
from torch import nn

import simulation
from simulation import World
from models import CnnAi, EquivariantAi, LinearAi

import datetime
import os

from matplotlib import pyplot as plt

script_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('start time:', script_start)

n_worlds = 128
episodes_count = 1500
learning_rate = 0.00001
world_width = 7
world_height = 7
max_steps = 5_000
reward_decrease_factor = 0.98
# max_reward_decrease_factor = 0.995
# reward_decrease_increaser = (max_reward_decrease_factor - reward_decrease_factor) / episodes_count * 3
n_train_episodes = 1

device = torch.device('cuda:0')
world = World(world_width, world_height, n_worlds, device)
# model = CnnAi(world)
# model = EquivariantAi(world)
model = LinearAi(world)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

model.load_state_dict(torch.load('./model_output/model_2021-06-26 13:01:56_19'))
optimizer.load_state_dict(torch.load('./model_output/optimizer_2021-06-26 13:01:56_19'))

plt.figure(0)
os.makedirs('./graph_output/', exist_ok=True)
os.makedirs('./model_output/', exist_ok=True)
losses = []
rewards = []

def running_avg(ls):
    result = []
    for end in range(1, len(ls) + 1):
        start = max(0, end - 10)
        total = 0
        for i in range(start, end):
            total += ls[i]
        result.append(total / (end - start))

    return result

for episode_nr in range(episodes_count):
    print('start episode', episode_nr)
    world = World(world_width, world_height, n_worlds, device)

    experience = []

    model.eval()

    total_reward = torch.zeros(n_worlds, device=device)

    n_steps = 0

    with torch.no_grad():
        while not torch.all(world.dead).item():
            alive = torch.logical_not(world.dead)
            network_input = torch.zeros(n_worlds, simulation.num_channels + 1, world.width + 6, world.height + 6, device=device)[alive]
            network_input[:, simulation.num_channels] = 1
            network_input[:, :-1, 3:-3, 3:-3] = world.space.to(torch.float)[alive]
            predicted_rewards = model(network_input)

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
            actions_weight += torch.randn(world.num_worlds, 4, device=device) * 0.03


            # actions_weight += torch.randn(4, world.num_worlds, device=device) * model.temperature
            randomize = torch.rand(world.num_worlds, device=device) < model.temperature
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

            experience.append((network_input, alive, reward, taken_action_idx, torch.max(predicted_rewards, dim=1).values))

            # print(len(experience), ':', torch.sum(world.dead))

    learn_scale = 1000 / n_steps
    print('n_steps =', n_steps)
    
    max_reward = torch.max(total_reward, 0).values.item()
    avg_reward = torch.sum(total_reward).item() / world.num_worlds
    print('\tmax reward:', max_reward)
    print('\tavg reward:', avg_reward)
    rewards.append(avg_reward)

    model.train()

    total_loss = 0
    steps = 0

    goals = [torch.zeros(0)] * (len(experience)) + [torch.zeros(n_worlds, dtype=torch.float, device=device)]
    _, alive_end, reward_end, _, _ = experience[-1]
    reached_end = [torch.zeros(0)] * len(experience) + [alive_end]

    for turn_nr in range(len(experience) - 1, -1, -1):
        _, alive, has_reward, _, _ = experience[turn_nr]
        goals[turn_nr] = goals[turn_nr + 1] * reward_decrease_factor + has_reward
        reached_end[turn_nr] = torch.logical_and(reached_end[turn_nr + 1], torch.logical_not(has_reward))
        
        # goals.append(goals[-1] * reward_decrease_factor + has_reward)

    # print(torch.sum(goals[0]) / n_worlds)
    
    for _ in range(n_train_episodes):
        for i in range(len(experience)):
            network_input, alive, reward, taken_action_idx, _ = experience[i]
            # _, next_alive, _, _, max_predicted_next = experience[i + 1]

            # next_reward = torch.zeros(n_worlds, device=device)
            # next_reward[next_alive] = max_predicted_next
            # next_reward = next_reward[alive]
            # next_reward += reward[alive]

            select_for_learning_large = torch.logical_and(torch.logical_not(reached_end[i]), alive)
            if torch.sum(select_for_learning_large).item() == 0:
                continue
            select_for_learning_small = select_for_learning_large[alive]
            selected_network_input = network_input[select_for_learning_small]

            goal = goals[i][select_for_learning_large] # (goals[-(i + 1)][alive] > 0).to(torch.float)

            optimizer.zero_grad()

            # print(torch.arange(0, network_input.shape[0]), taken_action_idx)
            predicted_rewards = model(selected_network_input)[
                torch.arange(0, selected_network_input.shape[0]),
                taken_action_idx[select_for_learning_large]
            ]

            loss = predicted_rewards - goal
            loss = torch.sum(loss * loss)
            scaled_loss = loss * learn_scale
            scaled_loss.backward()

            optimizer.step()
            
            total_loss += loss.item()
            steps += selected_network_input.shape[0]
            optimizer.step()
            
            total_loss += loss.item()
            steps += selected_network_input.shape[0]

        avg_loss = None if steps == 0.0 else total_loss / steps
        print('loss =', avg_loss)
        losses.append(avg_loss)

    # reward_decrease_factor = min(reward_decrease_factor + reward_decrease_increaser, max_reward_decrease_factor)

    if (episode_nr + 1) % 10 == 0:
        now = datetime.datetime.now()
        torch.save(model.state_dict(), 'model_output/model_' + script_start + '_'  + str(episode_nr))
        torch.save(optimizer.state_dict(), 'model_output/optimizer_' + script_start + '_'  + str(episode_nr))

        plt.plot(rewards)
        plt.plot(running_avg(rewards))

        plt.savefig('graph_output/avg_reward_' + script_start)
        plt.clf()

        plt.plot(losses)
        plt.plot(running_avg(losses))

        plt.savefig('graph_output/avg_loss_' + script_start)
        plt.clf()

        if episode_nr < 72:
            lr_scheduler.step()

        




