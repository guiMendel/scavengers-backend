import random
import os
import numpy as np
from collections import deque
from time import time
import torch
from pprint import pprint
from torch import nn
from time import time


# Comment this to enable GPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning

# === PARAMETERS

# Discount factor (used with bootstrapping)
gamma = 0.99

# Exploration rate
epsilon = 1.0
# How much it decays each step
epsilon_decay = 0.995
# The lowest value it will assume
epsilon_min = 0.1

# Learning rate of neural network
learning_rate = 0.001

# Get device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size, batch_size):
        super(NeuralNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size

        # Function to flatten input data
        self.flatten = nn.Flatten()

        # Layer stacks
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            # nn.Softmax(dim=1)
        )

        # Function to use for loss computation
        self.get_loss = nn.CrossEntropyLoss()

        # Get optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.from_numpy(x).to(torch.float32)
        x = self.flatten(x)

        return self.linear_relu_stack(x)

    def train_batch(self, x, y):
        # Get prediction
        prediction = self(x)

        # Get loss
        y = torch.from_numpy(y).to(torch.float32)
        loss = self.get_loss(prediction, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 




class Agent:
    def __init__(self, id, state_size, action_size, batch_size, model_name):
        self.birth_time = time()
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name

        # Will memorize experiences
        self.memory = {"states": deque(maxlen=2000), "actions": deque(maxlen=2000), "rewards": deque(maxlen=2000), "next_states": deque(maxlen=2000)}

        # These fields will store state-action pairs
        self.last_state = None
        self.last_action = None

        # Start the model
        self._build_model()

        # Load it's params
        try:
            self.model.load_state_dict(torch.load(self._get_path()))
        except (FileNotFoundError):
            print(f"{self.id} had no stored model")
            pass

    # Save model on destruction
    def __del__(self):
        print(f"{self.id} saving it's model and exiting. Total lifetime: {time() - self.birth_time} seconds")
        self.save_model()

    # Mounts and returns the keras model of this agent
    def _build_model(self):
        # Create new model
        self.model = NeuralNetwork(self.state_size, self.action_size, self.batch_size)

    # Train the model based on random samples of memory
    def _train(self):
        # Get indices for a batch
        indices = random.sample(range(len(self.memory["states"])), self.batch_size)

        states = np.reshape(np.array(self.memory["states"])[indices], [self.batch_size, -1])
        actions = np.reshape(np.array(self.memory["actions"])[indices], [self.batch_size, -1])
        rewards = np.reshape(np.array(self.memory["rewards"])[indices], [self.batch_size, -1])
        next_states = np.reshape(np.array(self.memory["next_states"])[indices], [self.batch_size, -1])

        # Get the value for (state, action) based on this observation
        before = time()
        
        next_state_values = self.model(next_states).detach().numpy()

        # print("Predict 1 Took", (time() - before) * 1000, "milliseconds")

        # TD(0) Bootstrap
        targets = [rewards[index] + self.gamma * np.amax(next_state_values[index]) for index in range(self.batch_size)]

        # Get what the current prediction for these states is
        before = time()

        targets_f = self.model(states).detach().numpy()

        # print("Predict 2 Took", (time() - before) * 1000, "milliseconds")

        # Update the predicted target with the observed targets
        for index in range(self.batch_size):
            targets_f[index][actions[index]] = targets[index]

        before = time()

        # Fit to new observation
        self.model.train_batch(states, targets_f)

        # print("Train Took", (time() - before) * 1000, "milliseconds")

        # validation_loss, validation_accuracy = self.model.evaluate(states, targets_f)

        # print("Loss: ", validation_loss, "Accuracy: ", validation_accuracy)

        # Discount epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    # Gets a model path from a model name
    def _get_path(self):
        return f"saved_models/{self.model_name}::{self.id}.pt"

    # Registers the results from last action while getting the next action based on the new state
    def iterate(self, step) -> int:
        (state, reward, terminal) = (step["state"], step["reward"], "terminal" in step)

        # pprint(state)

        # Transform state from vision matrix to single line
        state = np.reshape(state, [1, self.state_size]).tolist()

        # If not terminal and stored state-action pair, register result
        if not terminal and (self.last_state, self.last_action) != (None, None):
            self.register_results(reward, state)

        # Get new action
        return self.action_for(state)

    # Get the action index best suited for this state
    def action_for(self, state) -> int:
        # Store this state for later training
        self.last_state = state

        # Reshape state
        state = np.reshape(state, [1, self.state_size])

        # Random action chance
        if np.random.rand() <= self.epsilon:
            self.last_action = random.randrange(self.action_size)

        else:
            # Get current action values
            before = time()

            action_values = self.model(state)

            # print("Other Predict Took", (time() - before) * 1000, "milliseconds")

            # Get the one with best predicted return (and store it)
            self.last_action = np.argmax(action_values.detach().numpy()[0])

        return self.last_action

    # Register what the resulting state & reward was for last state-action encountered
    def register_results(self, reward, next_state):
        # Should have no store state-action pair
        if (self.last_state, self.last_action) == (None, None):
            raise Exception(f"Agent {self.id} tried registering results for state-action pair but the state-action pair was never registered")

        self.memory["states"].append(self.last_state)
        self.memory["actions"].append(self.last_action)
        self.memory["rewards"].append(reward)
        self.memory["next_states"].append(next_state)

        # Erase stored state-action pair
        (self.last_state, self.last_action) = (None, None)

        # Execute a training step
        if len(self.memory["states"]) >= self.batch_size:
            self._train()

    # Saves the current model's parameters
    def save_model(self):
        torch.save(self.model.state_dict(), self._get_path())
