import torch
import simulation
from simulation import World
from models import CnnAi, EquivariantAi
from time import sleep

world_width = 7
world_height = 7
n_worlds = 1

def wait(): sleep(0.3)

if __name__ == "__main__":
    device = torch.device('cuda:0')

    world = World(world_width, world_height, n_worlds, device)
    model = EquivariantAi(world)
    model.load_state_dict(torch.load('./model_output/best_model_eqv_2021-07-05_07:19:34_983'))
    model.temperature = 0.001

    model.eval()

    print('start evaluation')

    with torch.no_grad():
        print(world)
        wait()

        while not torch.all(world.dead).cpu():
            alive = torch.logical_not(world.dead)
            network_input = torch.zeros(n_worlds, simulation.num_channels + 1, world.width + 6, world.height + 6, device=device)[alive]
            network_input[:, simulation.num_channels] = 1
            network_input[:, :-1, 3:-3, 3:-3] = world.space.to(torch.float)[alive]
            network_input[:, -1, 3:-3, 3:-3] = 0
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
            actions_weight += torch.randn(world.num_worlds, 4, device=device) * 0.01


            # actions_weight += torch.randn(4, world.num_worlds, device=device) * model.temperature
            randomize = torch.rand(world.num_worlds, device=device) < model.temperature
            actions_weight[randomize] = torch.rand(world.num_worlds, 4, device=device)[randomize]
            actions_weight[impossible_large] = -100

            taken_action_idx = torch.argmax(actions_weight, dim=1)

            dx = model.actions_x[taken_action_idx]
            dy = model.actions_y[taken_action_idx]


            world.step(dx, dy)
            print(world)
            # input()
            wait()
