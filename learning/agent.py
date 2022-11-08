import random
import os
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from time import time

# Comment this to enable GPU usage
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Built based on https://www.dominodatalab.com/blog/deep-reinforcement-learning
# Optimize with https://www.tensorflow.org/guide/gpu_performance_analysis

# === PARAMETERS

# Discount factor (used with bootstrapping)
gamma = 0.95

# Exploration rate
epsilon = 1.0
# How much it decays each step
epsilon_decay = 0.995
# The lowest value it will assume
epsilon_min = 0.01

# Learning rate of neural network
learning_rate = 0.001


class Agent:
    def __init__(self, id, state_size, action_size, batch_size):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Will memorize experiences
        self.memory = {"states": deque(maxlen=2000), "actions": deque(maxlen=2000), "rewards": deque(maxlen=2000), "next_states": deque(maxlen=2000)}

        # These fields will store state-action pairs
        self.last_state = None
        self.last_action = None

        # Start the model
        self._build_model()

    # Mounts and returns the keras model of this agent
    def _build_model(self):
        # Create new model
        self.model = Sequential()

        # Add first hidden layer
        self.model.add(Dense(32, activation=tf.nn.relu, input_dim=self.state_size))
        # Add another hidden layer
        self.model.add(Dense(32, activation=tf.nn.relu))

        # Output layer (output the z values directly from the neurons with the "linear" activation)
        self.model.add(Dense(self.action_size, activation="linear"))

        # Compile it
        # According to the guide, "mean squared error is an appropriate choice of cost function when we use linear activation in the output layer"
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])

    # Train the model based on random samples of memory
    def _train(self):
        # Get indices for a batch
        indices = random.sample(range(len(self.memory["states"])), self.batch_size)

        states = np.reshape(np.array(self.memory["states"])[indices], [self.batch_size, -1])
        actions = np.reshape(np.array(self.memory["actions"])[indices], [self.batch_size, -1])
        rewards = np.reshape(np.array(self.memory["rewards"])[indices], [self.batch_size, -1])
        next_states = np.reshape(np.array(self.memory["next_states"])[indices], [self.batch_size, -1])

        # Get the value for (state, action) based on this observation
        next_state_values = self.model.predict(next_states, verbose=0)

        # TD(0) Bootstrap
        targets = [rewards[index] + self.gamma * np.amax(next_state_values[index]) for index in range(self.batch_size)]

        # Get what the current prediction for these states is
        targets_f = self.model.predict(states, verbose=0)

        # Update the predicted target with the observed targets
        for index in range(self.batch_size):
            targets_f[index][actions[index]] = targets[index]

        before = time()

        # Fit to new observation
        self.model.fit(states, targets_f, epochs=1, verbose=0)

        print("Took", (time() - before) * 1000, "milliseconds")

        # validation_loss, validation_accuracy = self.model.evaluate(states, targets_f)

        # print("Loss: ", validation_loss, "Accuracy: ", validation_accuracy)

        # Discount epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon_min *= self.epsilon_decay

    # Registers the results from last action while getting the next action based on the new state
    def iterate(self, step) -> int:
        (state, reward, terminal) = (step["state"], step["reward"], "terminal" in step)

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
            action_values = self.model.predict(state, verbose=0)

            # Get the one with best predicted return (and store it)
            self.last_action = np.argmax(action_values[0])

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
        self.model.save_weights(f"{id}.model")
