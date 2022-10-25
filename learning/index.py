from learning.agent import Agent

# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning

# === HYPER PARAMETERS

# Only agent's coordinates as state for now
state_size = 2

# Only actions "random_move" and "idle" for now
action_size = 2

# Training session parameters
batch_size = 32


# Maps each action label to an index
actions = [
    "idle",
    "move"
]

# Generate new agent
def new_agent(id: str):
    return Agent(id, state_size, action_size, batch_size)
