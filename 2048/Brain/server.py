from flask import Flask, request
import json
import numpy as np
import tensorflow as tf
from agent import DQNAgent
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
    state = parse_state(payload['state'])

    with graph.as_default():

        # Invokes Cloud ML model
        response = {"action" : ACTIONS[agent.act(state)]}

    return json.dumps(response)


@app.route('/report_results', methods=['POST'])
def handle_results():
    global rounds_played

    payload = json.loads(request.data)[0]
   #print(payload)

    state = parse_state(payload['current_state'])
    next_state = parse_state(payload['new_state'])
    reward = float(payload['reward'] > 0)
    done = payload['done']
    action = ACTIONS_MAP[payload['action']]

    agent.remember(state=state, next_state=next_state, action=action, reward=reward, done=done)

    rounds_played += 1
    #print("played ", str(rounds_played), " rounds")
    if (rounds_played > 5) or done:
        #print("\n\ntraining!!\n\n")
        with graph.as_default():
            pass
            agent.replay(500)
        rounds_played = 0
        print(f'epsilon={agent.epsilon}')
    return json.dumps({'result': 'success!'})


def parse_state(state):
    x = np.concatenate(state).ravel().tolist()
    x = np.array(x).reshape(1, 16)
    x[x == 0] = 1
    x = np.log2(x) / 16.0
    return x

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)