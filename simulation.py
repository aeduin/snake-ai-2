import torch

num_channels = 4
snake_channel = 0
snake_head_channel = 1
food_channel = 2
empty_channel = 3

space_type = torch.int16

class World():
    def __init__(self, width: int, height: int, num_worlds: int, device):
        self.device = device
        self.size = torch.Tensor([width, height]).to(device)

        center_x = width // 2
        center_y = width // 2
        self.snake_head_x = torch.reshape(torch.Tensor([center_x]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.snake_head_y = torch.reshape(torch.Tensor([center_y]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.snake_size = torch.reshape(torch.Tensor([2]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.dead = torch.reshape(torch.Tensor([False]), (1, 1)).to(device, bool).repeat(num_worlds, 1)

        self.space = torch.zeros((num_worlds, num_channels, width, height)).to(device, space_type)

        self.space[:, snake_head_channel, self.snake_head_x, self.snake_head_y] = 1
        self.space[:, snake_channel, self.snake_head_x, self.snake_head_y : self.snake_head_y + 2] = 2
        # self.space[:, snake_channel, self.snake_head_x, self.snake_head_y + 1] = 1
        
        # self.place_food(1)

        self.space[:, food_channel, center_x + 3, center_y] = 1

    def step(self, move_x: torch.Tensor, move_y: torch.Tensor):
        # Calculate the new position of the head
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(space_type))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(space_type))
        
        self.dead = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(
                self.dead,
                new_head_x < 0),
                new_head_x >= self.size[0]),
                new_head_y < 0),
                new_head_y >= self.size[1]),
                self.space[:, snake_channel, new_head_x, new_head_y] > 0)

        # Recalculate the new position of the head in case it died
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(space_type))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(space_type))

        # print(new_head_x, new_head_y)

        # It there is food at the new location of the head, the snake eats the food
        eat_food = (self.space[:, food_channel, new_head_x, new_head_y] == 1).to(space_type).view(-1)

        # The time until a piece of the snake disappears decreases by 1 if it has eaten no food, or by 0 if it has
        reduce_time = eat_food - 1
        decrease_time = torch.logical_not(eat_food)
        
        print(decrease_time)
        print(eat_food)
        # print(reduce_time.shape)
        # print(self.space[decrease_time][snake_channel].shape)
        # print((self.space[decrease_time][snake_channel, :, :] > 0).shape)
        selected = self.space[decrease_time, snake_channel, :, :] > 0
        print(selected)
        self.space[snake_channel, selected] -= 1
        
        # Add 1 to the snake length if it has eaten food
        self.snake_size += eat_food
        
        # Move snake
        self.space[:, snake_channel, new_head_x, new_head_y] = self.snake_size.to(space_type)
        self.space[:, snake_head_channel, new_head_x, new_head_y] = 1
        self.space[:, snake_head_channel, self.snake_head_x, self.snake_head_y] = 0

        # Move food
        self.space[:, food_channel, new_head_x, new_head_y] = 0

        # self.place_food(eat_food)
        
        # Update where the snake head is located
        self.snake_head_x = new_head_x
        self.snake_head_y = new_head_y

    def place_food(self, new_food_required):
        valid_idx = torch.nonzero(self.space[:, :, snake_channel] == 0)
        choice = torch.randint(0, valid_idx.shape[0], (1,))
        # print('choice:', choice)
        # print('valid_idx.shape:', valid_idx.shape)
        coordinate = valid_idx[choice[0]]
        
        # print('coordinate:', coordinate)
        
        self.space[coordinate[0], coordinate[1], food_channel:food_channel+1] += new_food_required

    def __str__(self):
        result = ""

        for y in range(int(self.size[1])):
            for x in range(int(self.size[0])):
                here = self.space[0, x, y]
                if here[snake_head_channel] > 0:
                    result += "h"
                elif here[snake_channel] > 0:
                    result += "s"
                elif here[food_channel] > 0:
                    result += "f"
                elif (x + y) % 2 == 0:
                    result += " "
                else:
                    result += "."
            result += "\n"

        return result

if __name__ == "__main__":
    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    w = World(10, 10, 2, device)

    # for i in range(1000):
    #     for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    #         w.step(dx, dy)
    for i in range(5):
        w.step(1, 0)
        print(w)
        input()
