from flask import Flask, request
import json
import numpy as np
import tensorflow as tf
VERSION_NAME = 'v2'
MODEL_NAME = 'transportation_mode'
PROJECT_ID = 'gad-playground-212407'
from agent import DQNAgent

app = Flask(__name__)

# Initialize objects
agent = DQNAgent(state_size=(4,4), action_size=4)
global graph
graph = tf.get_default_graph()

ACTIONS = ['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight']
ACTIONS_MAP = {'ArrowDown':0, 'ArrowUp':1, 'ArrowLeft':2, 'ArrowRight':3}

rounds_played = 0

@app.route('/act', methods=['POST'])
def act():
    """
    Request handler
    :return:
    """
    #print (request.data)
    payload = json.loads(request.data)
    state = np.concatenate(payload['state']).ravel().tolist()
    state = np.array(state).reshape(1, 16)

    with graph.as_default():

        # Invokes Cloud ML model
        response = {"action" : ACTIONS[agent.act(state)]}

    return json.dumps(response)


@app.route('/report_results', methods=['POST'])
def handle_results():
    global rounds_played

    payload = json.loads(request.data)[0]
    print(payload)

    state = np.concatenate(payload['current_state']).ravel().tolist()
    state = np.array(state).reshape(1,16)
    next_state = np.concatenate(payload['new_state']).ravel().tolist()
    next_state = np.array(next_state).reshape(1,16)
    reward = payload['reward']
    done = payload['done']
    action = ACTIONS_MAP[payload['action']]

    agent.remember(state=state, next_state=next_state, action=action, reward=reward, done=done)

    rounds_played += 1
    print("played ", str(rounds_played), " rounds")
    if (rounds_played > 10) or done:
        print("\n\ntraining!!\n\n")
        with graph.as_default():
            agent.replay(5)
        rounds_played = 0

    return json.dumps({'result': 'success!'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)