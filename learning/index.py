from learning.agent import Agent

# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning

# === HYPER PARAMETERS

# Size of agent's view range, in unit count (breadth * range * channels of vision)
state_shape = (3, 21, 21)

# Training session parameters
batch_size = 32

# Maps each action label to an index
actions = [
    "idle",
    "tag",
    "move_ahead",
    "move_right",
    "move_left",
    "move_back",
    "face_right",
    "face_left"
]

# Generate new agent
def new_agent(id: str, model_name: str):
    return Agent(id, state_shape, len(actions), batch_size, model_name)
