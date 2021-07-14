# Snake AI
This is the source code for my bacholor thesis project.

# Build instructions:
- Install miniconda or anaconda
- Clone this repository
- Run `conda env create -f ./environment.yml` in this repository to create the environment `aeduin-rl`
- Run `conda activate aeduin-rl`
- Install [https://github.com/adambielski/GrouPy]. Make sure you install it inside the `aeduin-rl` environment by running the command above if you open an new shell.

# Running the scripts
- Execute `python train.py` to train an AI. Change the `model_name` variable in the source code to change which network architecture is used.
- If you want to see it play: in `visualize_play.py`, modify the `model = ` and `model.load_state_dict(...)` variables (`train.py` automatically outputs trained state dicts in `model_output/`). Then run `python visualize_play.py`
