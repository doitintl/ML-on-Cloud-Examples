import logging
import requests
logging.basicConfig(level=logging.DEBUG)
import json
import numpy as np

from Brain.agent import DQNAgent

def test_model_creation():
    # Arrange
    print("great success")

    agent = DQNAgent(state_size=(4, 4), action_size=4)

    model = agent.model
    X = np.ones((1,16))
    model.predict(X)
    print(model.summary())
    print("great success")
    assert True

def test_model_act():
    # Arrange
    print("great success")

    agent = DQNAgent(state_size=(4, 4), action_size=4)
    X = np.ones((1,16))

    result = agent.act(X)
    print("great success")
    assert True