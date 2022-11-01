from learning.agent import Agent

# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning

# === HYPER PARAMETERS

# Size of agent's view range, in unit count
state_size = 81

# Training session parameters
batch_size = 32

# Maps each action label to an index
actions = [
    "idle",
    "move_ahead"
]

# Generate new agent
def new_agent(id: str):
    return Agent(id, state_size, len(actions), batch_size)
