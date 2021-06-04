import torch

num_channels = 4
snake_channel = 0
snake_head_channel = 1
food_channel = 2
empty_channel = 3

space_type = torch.int16

class World():
    def __init__(self, width: int, height: int, num_worlds: int, device: torch.device):
        self.device = device
        self.size = torch.tensor([width, height]).to(device)
        self.width = width
        self.height = height
        self.num_worlds = num_worlds

        self.all_worlds_tensor = torch.arange(0, num_worlds, 1, dtype=torch.long, device=device)
        self.all_width_tensor  = torch.arange(0, width,      1, dtype=torch.long, device=device)
        self.all_height_tensor = torch.arange(0, height,     1, dtype=torch.long, device=device)

        center_x = width // 2
        center_y = width // 2
        self.snake_head_x = torch.tensor([center_x - 1]).to(device, torch.long).repeat(num_worlds)
        self.snake_head_y = torch.tensor([center_y]).to(device, torch.long).repeat(num_worlds)
        self.snake_size = torch.tensor([2]).to(device, torch.long).repeat(num_worlds)
        self.dead = torch.tensor([False]).to(device, torch.bool).repeat(num_worlds)
        # print('self.snake_head', self.snake_head_x, self.snake_head_y)

        self.space = torch.zeros((num_worlds, num_channels, width, height)).to(device, space_type)

        # print('space = ', self.space[:, snake_channel, self.snake_head_x, self.snake_head_y - 2])
        # print('diag1 =', torch.diagonal(self.space[self.all_worlds_tensor, snake_channel, self.snake_head_x, self.snake_head_y - 1]))
        self.space[self.all_worlds_tensor, snake_head_channel, self.snake_head_x, self.snake_head_y] = 1
        self.space[self.all_worlds_tensor, snake_channel,      self.snake_head_x, self.snake_head_y] = 2
        self.space[self.all_worlds_tensor, snake_channel,      self.snake_head_x - 1, self.snake_head_y] = 1
        # self.space[:, snake_channel, self.snake_head_x, self.snake_head_y + 1] = 1
        # self.place_food(1)

        self.space[:, food_channel, center_x + 3, center_y] = 1

    def step(self, move_x: torch.Tensor, move_y: torch.Tensor):
        print('self.dead 1', self.dead)
        # Calculate the new position of the head
        # print('dead', self.dead)
        # print('snake_head_x', self.snake_head_x)
        # print('move_x', move_x)
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(space_type))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(space_type))
        # print('new_head1', new_head_x, new_head_y)
 
        # print('space', torch.diag(self.space[:, snake_channel, new_head_x, new_head_y]) > 0)
        self.dead = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(
                self.dead,
                new_head_x < 0),
                new_head_x >= self.size[0]),
                new_head_y < 0),
                new_head_y >= self.size[1])
        print('self.dead 2', self.dead)

        alive = torch.logical_not(self.dead)
        # print((self.space[alive, snake_channel, new_head_x[alive], new_head_y[alive]]) > 0)

        # Die if the snake hits itself on the new location
        print('going to die:', new_head_x[alive], new_head_y[alive])
        self.dead[alive] = (self.space[alive, snake_channel, new_head_x[alive], new_head_y[alive]]) > 0
        alive = torch.logical_not(self.dead)
        print('self.dead 3', self.dead)

        # Recalculate the new position of the head in case it died
        new_head_x = (self.snake_head_x + move_x * (1 - self.dead.to(space_type)))
        new_head_y = (self.snake_head_y + move_y * (1 - self.dead.to(space_type)))
        # print('dead2', self.dead)
        # print('snake_head_x2', self.snake_head_x)
        # print('move_x2', move_x)

        # print(new_head_x, new_head_y)
        # print('new_head', new_head_x, new_head_y)

        print('alive', alive)
        # It there is food at the new location of the head, the snake eats the food
        eat_food = torch.logical_and(
            self.space[self.all_worlds_tensor, food_channel, new_head_x, new_head_y] == 1,
            alive
        ).to(torch.bool) # .view(-1)

        # The time until a piece of the snake disappears decreases by 1 if it has eaten no food, or by 0 if it has
        # reduce_time = eat_food - 1
        should_decrease_tail = torch.logical_not(eat_food)
        
        print('should_decrease_tail', should_decrease_tail)
        print('eat_food', eat_food)
        # print(reduce_time.shape)
        # print(self.space[should_decrease_tail][snake_channel].shape)
        # print((self.space[should_decrease_tail][snake_channel, :, :] > 0).shape)
        decrease_snake_tail = torch.zeros((self.num_worlds, num_channels, self.width, self.height), device=self.device, dtype=torch.bool)
        # print('decrease_snake_tail.shape', decrease_snake_tail.shape)
        all_snake_channels = torch.tensor([snake_channel]).repeat(self.num_worlds)

        
        decrease_snake_tail[:, snake_channel] = self.space[
            :,
            snake_channel,
            :,
            :
        ] > 0

        decrease_snake_tail[torch.logical_or(eat_food, self.dead), : , :, :] = 0
        # print(decrease_snake_tail)
        print('space shape', self.space.shape)
        print('decrease_snake_tail', decrease_snake_tail[:, snake_channel, : ,:])
        self.space[decrease_snake_tail] -= 1
        
        # Add 1 to the snake length if it has eaten food
        print(self.snake_size)
        print(eat_food)
        self.snake_size += eat_food.to(space_type) # add 1 if food is eaten
        
        # Move snake
        self.space[alive, snake_channel, new_head_x[alive], new_head_y[alive]] = self.snake_size.to(space_type)[alive]
        self.space[alive, snake_head_channel, new_head_x[alive], new_head_y[alive]] = 1
        self.space[alive, snake_head_channel, self.snake_head_x[alive], self.snake_head_y[alive]] = 0

        # Move food
        self.space[self.all_worlds_tensor, food_channel, new_head_x, new_head_y] = 0

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
        result = "+" + ("-" * int(self.size[1])) + "+\n"
        
        for batch in range(self.num_worlds):
            for y in range(int(self.size[1])):
                result += "|"
                for x in range(int(self.size[0])):
                    here = self.space[batch, :, x, y]
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
                result += "|\n"
            result += "+" + ("-" * int(self.size[1])) + "+\n"

        return result

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    num_worlds = 3

    w = World(10, 10, num_worlds, device)

    # for i in range(1000):
    #     for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    #         w.step(dx, dy)
    print(w)
    input()
    for i in range(7):
        w.step(torch.tensor([1, 0, 0], device=device).to(torch.long), torch.tensor([0, 1, -1], device=device).to(torch.long))
        print(w)
        input()
