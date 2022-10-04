import random
import os
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# === PARAMETERS

# Discount factor (used with bootstrapping)
gama = 0.95

# Exploration rate
epsilon = 1.0
# How much it decays each step
epsilon_decay = 0.995
# The lowest value it wil assume
epsilon_min = 0.01

# Learning rate of neural network
learning_rate = 0.001


class Agent:
    def _init_(self, id, state_size, action_size, batch_size):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gama = gama
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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
        self.model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

    # Train the model based on random samples of memory
    def _train(self):
        # Get a batch
        minibatch = random.sample(self.memory, self.batch_size)

        # For each entry in batch
        for state, action, reward, next_state, done in minibatch:
            # Get the value for (state, action) based on this observation
            target = reward

            # TD(0) Bootstrap if not terminal state
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # Get what the current prediction for this state is
            target_f = self.model.predict(state)

            # Apply the observed target
            target_f[0][action] = target

            # Fit to new observation
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # Discount epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon_min *= self.epsilon_decay

    # Get the action index best suited for this state
    def action_for(self, state):
        # Store this state for later training
        self.last_state = state

        # Random action chance
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Get current action values
        action_values = self.model.predict(state)

        # Get the one with best predicted return (and store it)
        self.last_action = np.argmax(action_values[0])

        return self.last_action

    # Register what the resulting state & reward was for last state-action encountered
    def register_results(self, reward, next_state, done):
        # Should have no store state-action pair
        if (self.last_state, self.last_action) != (None, None):
            raise Exception(f"Agent {self.id} tried registering results for state-action pair but the state-action pair was never registered")

        self.memory.append((self.last_state, self.last_action, reward, next_state, done))

        # Erase stored state-action pair
        (self.last_state, self.last_action) = (None, None)

        # Execute a training step
        if len(self.memory) >= self.batch_size:
            self._train()

    # Saves the current model's parameters
    def save_model(self):
        self.model.save_weights(f"{id}.model")
