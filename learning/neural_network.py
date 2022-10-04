# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning

# === HYPER PARAMETERS

# Only agent's coordinates as state for now
state_size = 2

# Only actions "random_move" and "idle" for now
action_size = 2

# Training session parameters
batch_size = 32
n_episodes = 1000