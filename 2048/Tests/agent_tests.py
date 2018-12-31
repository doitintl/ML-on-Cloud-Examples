import logging
import requests
logging.basicConfig(level=logging.DEBUG)
import json
import numpy as np

def test_act():
    '''
    :return:
    '''

    # Arrange
    state = np.random.randint(0,256, size=(4,4)).tolist()
    r = requests.post('http://localhost:8080/act', json={'state': state})
    print (r.text)
    assert (True)


def test_report():
    '''
    :return:
    '''
    sample = json.loads("""{"current_state":
    [[0,2,8,0],[4,16,32,0],[16,2,8,0],[8,8,2,0]],
    "action":"ArrowRight",
    "reward":0,
    "new_state":[[0,2,8,0],[4,16,32,0],[16,2,8,0],[8,8,2,0]],"done":false}""")

    # Arrange
    state = np.random.randint(0,256, size=(4,4)).tolist()
    r = requests.post('http://localhost:8080/report_results', json={'state': state})
    print (r.text)
    assert (True)