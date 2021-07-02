import torch

num_channels = 3
snake_channel = 0
snake_head_channel = 1
food_channel = 2

space_type = torch.int16

initial_snake_size = 3

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
        self.snake_head_x = torch.tensor([center_x]).to(device, torch.long).repeat(num_worlds)
        self.snake_head_y = torch.tensor([center_y]).to(device, torch.long).repeat(num_worlds)
        self.snake_size = torch.tensor([initial_snake_size]).to(device, torch.long).repeat(num_worlds)
        self.dead = torch.tensor([False]).to(device, torch.bool).repeat(num_worlds)

        self.space = torch.zeros((num_worlds, num_channels, width, height)).to(device, space_type)

        self.space[self.all_worlds_tensor, snake_head_channel, self.snake_head_x, self.snake_head_y] = 1
        self.space[self.all_worlds_tensor, snake_channel,      self.snake_head_x, self.snake_head_y] = initial_snake_size
        # self.space[self.all_worlds_tensor, snake_channel,      self.snake_head_x - 1, self.snake_head_y] = 1
        # self.space[:, snake_channel, self.snake_head_x, self.snake_head_y + 1] = 1
        # self.place_food(1)

        # self.space[:, food_channel, center_x + 3, center_y] = 1
        self.place_food(torch.ones(num_worlds, dtype=torch.bool, device=device))


    def step(self, move_x: torch.Tensor, move_y: torch.Tensor):
        # Calculate the new position of the head
        new_head_x = self.snake_head_x + move_x * (1 - self.dead.to(space_type))
        new_head_y = self.snake_head_y + move_y * (1 - self.dead.to(space_type))

        # print(new_head_x.to('cpu').numpy()[0])
 
        self.dead = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(
                self.dead,
                new_head_x < 0),
                new_head_x >= self.width),
                new_head_y < 0),
                new_head_y >= self.height)

        alive = torch.logical_not(self.dead)

        # Die if the snake hits itself on the new location
        self.dead[alive] = (self.space[alive, snake_channel, new_head_x[alive], new_head_y[alive]]) > 0
        alive = torch.logical_not(self.dead)

        # Recalculate the new position of the head in case it died
        new_head_x = (self.snake_head_x + move_x * (1 - self.dead.to(space_type)))
        new_head_y = (self.snake_head_y + move_y * (1 - self.dead.to(space_type)))

        # It there is food at the new location of the head, the snake eats the food
        eat_food = torch.logical_and(
            self.space[self.all_worlds_tensor, food_channel, new_head_x, new_head_y] == 1,
            alive
        ).to(torch.bool) # .view(-1)

        # The time until a piece of the snake disappears decreases by 1 if it has eaten no food, or by 0 if it has
        # reduce_time = eat_food - 1
        
        decrease_snake_tail = torch.zeros((self.num_worlds, num_channels, self.width, self.height), device=self.device, dtype=torch.bool)

        
        decrease_snake_tail[:, snake_channel] = self.space[
            :,
            snake_channel,
            :,
            :
        ] > 0

        decrease_snake_tail[torch.logical_or(eat_food, self.dead), : , :, :] = 0
        self.space[decrease_snake_tail] -= 1
        
        # Add 1 to the snake length if it has eaten food
        self.snake_size += eat_food.to(space_type) # add 1 if food is eaten
        
        # Move snake
        self.space[alive, snake_channel, new_head_x[alive], new_head_y[alive]] = self.snake_size.to(space_type)[alive]
        self.space[alive, snake_head_channel, new_head_x[alive], new_head_y[alive]] = 1
        self.space[alive, snake_head_channel, self.snake_head_x[alive], self.snake_head_y[alive]] = 0

        # Move food
        self.space[self.all_worlds_tensor, food_channel, new_head_x, new_head_y] = 0

        self.place_food(eat_food)
        
        # Update where the snake head is located
        self.snake_head_x = new_head_x
        self.snake_head_y = new_head_y

        snake_length = torch.sum(self.space[:, snake_channel, :, :] > 0, dim=[1, 2])
        self.dead = torch.logical_or(self.dead, snake_length >= self.width * self.height - 1)

        return eat_food

    def place_food(self, new_food_required):
        has_no_snake = self.space[:, snake_channel, :, :] == 0

        num_locations = has_no_snake.to(dtype=torch.long)
        num_locations = torch.sum(num_locations, dim=1)
        num_locations = torch.sum(num_locations, dim=1)

        selected_locations = (
            torch.rand((self.num_worlds,), device=self.device)
            *
            num_locations
        ).to(torch.long)
        selected_locations_cumulative = torch.cumsum(num_locations, 0) - num_locations + selected_locations

        selected_food = torch.zeros(self.space.shape, dtype=torch.bool)
        selected_food[:, food_channel, :, :] = has_no_snake

        copy_selected_space = self.space[selected_food]
        copy_selected_space[selected_locations_cumulative] += new_food_required        

        self.space[selected_food] = copy_selected_space 


        
        """
        valid_idx = torch.nonzero(torch.logical_not(has_no_snake))
        choice = torch.randint(0, valid_idx.shape[0], (1,))
        coordinate = valid_idx[choice[0]]
        
        self.space[coordinate[0], coordinate[1], food_channel:food_channel+1] += new_food_required
        """

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
    """
    num_worlds = 256


    w = World(10, 10, num_worlds, device)
    actions_x = torch.tensor([1], dtype=torch.long, device=device).repeat(num_worlds)
    actions_y = torch.tensor([0], dtype=torch.long, device=device).repeat(num_worlds)

    from time import perf_counter
    
    start = perf_counter()

    for i in range(16 * 64):
        w.step(actions_x, actions_y)

    w_space = w.space.cpu()

    print(perf_counter() - start)

    exit()
    """

    num_worlds = 4

    w = World(10, 10, num_worlds, device)

    # for i in range(1000):
    #     for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    #         w.step(dx, dy)
    print(w)
    input()
    for i in range(7):
        w.step(torch.tensor([1, 0, 0, 1], device=device).to(torch.long), torch.tensor([0, 1, -1, 0], device=device).to(torch.long))
        print(w)
        input()
