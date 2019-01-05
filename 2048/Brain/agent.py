from keras import layers as lyr
from keras.models import Model
import numpy as np
from collections import deque
import random

# Deep Q-learning Agent
class DQNAgent:
    ACTIONS_MAP = {'ArrowDown': 0, 'ArrowUp': 1, 'ArrowLeft': 2, 'ArrowRight': 3}

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.MEMORY_SIZE = 1024
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.001
        self.model = self._build_model()


    def _build_model(self):
        inputs = lyr.Input(shape=(4, 4, 1))

        conv_22 = lyr.Conv2D(16, (2, 2), padding='same', activation='relu')(inputs)
        pool_22 = lyr.MaxPooling2D((2, 2))(conv_22)

        conv_14 = lyr.Conv2D(16, (1, 4), padding='same', activation='relu')(inputs)
        pool_14 = lyr.MaxPooling2D((1, 2))(conv_14)

        conv_41 = lyr.Conv2D(16, (4, 1), padding='same', activation='relu')(inputs)
        pool_41 = lyr.MaxPooling2D((2, 1))(conv_41)

        flat_22 = lyr.Flatten()(pool_22)
        flat_14 = lyr.Flatten()(pool_14)
        flat_41 = lyr.Flatten()(pool_41)

        concat = lyr.concatenate([flat_22, flat_14, flat_41])

        dense_1 = lyr.Dense(32, activation='relu')(concat)
        dense_2 = lyr.Dense(32, activation='relu')(dense_1)

        output = lyr.Dense(4, activation='linear')(dense_2)

        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='mse',
                      optimizer='nadam')
        try:
            model.load_weights('weights.h5')
        except Exception as e:
            print (e)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.rotate(np.random.randint(0,len(self.memory),1)[0])
            self.memory.pop()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        # Replay
        inputs = []
        targets = np.zeros((batch_size, 4))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            inputs.append(state)
            old_q = self.model.predict(np.array([state]))[0]
            new_q = self.model.predict(np.array([next_state]))[0]
            update_target = np.copy(old_q)

            if done:
                update_target[action] = -1
            else:
                update_target[action] = reward + (self.gamma * np.max(new_q))
            targets[i] = update_target

        loss = self.model.train_on_batch(np.array(inputs), targets)# , epochs=5, verbose=2)
        print("loss = ", loss)
        self.model.save_weights('weights.h5', overwrite=True)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        elif np.random.rand() <= self.epsilon:
                self.epsilon += 0.03

