from keras import layers as lyr
from keras.models import Model
import numpy as np
from collections import deque
import random
import tensorflow as tf

# Deep Q-learning Agent
class DQNAgent:
    ACTIONS_MAP = {'ArrowDown': 0, 'ArrowUp': 1, 'ArrowLeft': 2, 'ArrowRight': 3}

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.99  # exploration rate
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.001
        self.learning_rate = 0.001
        self.model = self._build_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = lyr.Input(shape=(16, ))
        #flattern = lyr.Flatten()(input)
        dense_1 = lyr.Dense(64, activation='tanh')(input)
        dense_2 = lyr.Dense(64, activation='tanh')(dense_1)
        dense_3 = lyr.Dense(64, activation='relu')(dense_2)
        output = lyr.Dense(self.action_size, activation='linear')(dense_3)

        model = Model(inputs=input, outputs=output)
        model.compile(loss='mse',
                      optimizer='nadam')
        try:
            model.load_weights('weights.h5')
        except:
            pass

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        # Replay
        inputs = np.zeros((batch_size, 16))
        targets = np.zeros((batch_size, 4))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            inputs[i:i + 1] = state
            old_q = self.model.predict(state)[0]
            new_q = self.model.predict(next_state)[0]
            update_target = np.copy(old_q)

            if not done:
                update_target[action] = -1
            else:
                update_target[action] = reward + (self.gamma * np.max(new_q))
            targets[i] = update_target

        self.model.train_on_batch(inputs, targets)

        self.model.save_weights('weights.h5', overwrite=True)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
