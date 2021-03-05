

if __name__ == "__main__":
    from argparse import ArgumentParser
    from sys import argv

    arg_parser = ArgumentParser()
    arg_parser.add_argument("-t", "--train")

    args = arg_parser.parse_args(argv)

    print("start")

