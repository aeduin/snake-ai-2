import torch
from simulation import World
from models import RotatedAI

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    w = World(10, 10, 4, device)

    model = RotatedAI(w.size, device)
    model.eval()
    
    with torch.no_grad():
        print("start eval")
        for i in range(20):
            action_x, action_y = model.get_actions(w)
            w.step(action_x, action_y)

            # print(w)
            # input()

            if i == 0:
                print("done first eval")
            print(w)
            input()



if __name__ == "__main__":
    from argparse import ArgumentParser
    from sys import argv

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--train")

    args = arg_parser.parse_args(argv)

    print("start")

