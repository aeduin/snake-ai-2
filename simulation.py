import torch

num_channels = 4
snake_channel = 0
snake_head_channel = 1
food_channel = 2
empty_channel = 3

class World():
    def __init__(self, width: int, height: int, num_worlds: int, device):
        self.device = device
        self.size = torch.Tensor([num_worlds, width, height]).to(device)

        self.all_worlds_tensor = torch.arange(0, num_worlds, 1, dtype=torch.long, device=device)
        self.all_width_tensor  = torch.arange(0, width,      1, dtype=torch.long, device=device)
        self.all_height_tensor = torch.arange(0, height,     1, dtype=torch.long, device=device)

        center_x = width // 2
        center_y = width // 2
        self.snake_head_x = torch.reshape(torch.Tensor([center_x]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.snake_head_y = torch.reshape(torch.Tensor([center_y]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.snake_size = torch.reshape(torch.Tensor([2]), (1, 1)).to(device, torch.long).repeat(num_worlds, 1)
        self.dead = torch.reshape(torch.Tensor([False]), (1, 1)).to(device).repeat(num_worlds, 1)

        self.space = torch.zeros((num_worlds, num_channels, width, height)).to(device, torch.int16)

        self.space[self.all_worlds_tensor, snake_head_channel, self.snake_head_x, self.snake_head_y] = 1
        self.space[self.all_worlds_tensor, snake_channel, self.snake_head_x, self.snake_head_y - 1] = 2
        # self.space[self.all_worlds_tensor, snake_channel, self.snake_head_x, self.snake_head_y + 1] = 1
        
        # self.place_food(1)

        self.space[self.all_worlds_tensor, food_channel, center_x + 3, center_y] = 1

    def step(self, move_x: torch.Tensor, move_y: torch.Tensor):
        # Calculate the new position of the head
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(torch.long))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(torch.long))
        
        self.dead = self.dead or new_head_x < 0 or new_head_x >= self.size[1] or new_head_y < 0 or new_head_y >= self.size[2] or self.space[self.all_worlds_tensor, snake_channel, new_head_x, new_head_y] > 0

        # Recalculate the new position of the head in case it died
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(torch.long))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(torch.long))

        # print(new_head_x, new_head_y)

        # If there is food at the new location of the head, the snake eats the food
        eat_food = (self.space[self.all_worlds_tensor, food_channel, new_head_x, new_head_y] == 1).to(torch.long)

        # The time until a piece of the snake disappears decreases by 1 if it has eaten no food, or by 0 if it has
        reduce_time = eat_food - 1

        self.space[self.space[self.all_worlds_tensor, snake_channel, self.all_width_tensor, self.all_height_tensor] > 0] += reduce_time
        
        # Add 1 to the snake length if it has eaten food
        self.snake_size += eat_food
        
        # Move snake
        self.space[new_head_x, new_head_y, snake_channel] = self.snake_size.to(torch.int16)
        self.space[new_head_x, new_head_y, snake_head_channel] = 1
        self.space[self.snake_head_x, self.snake_head_y, snake_head_channel] = 0

        # Move food
        self.space[new_head_x, new_head_y, food_channel] = 0

        self.place_food(eat_food)
        
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
                here = self.space[x, y]
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
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    w = World(10, 10, 5, device)

    # for i in range(1000):
    #     for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    #         w.step(dx, dy)
    for i in range(5):
        w.step(torch.tensor([1], device=device), torch.tensor([0], device=device))
        print(w)
        input()
