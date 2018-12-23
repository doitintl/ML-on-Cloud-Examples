import logging
import requests
logging.basicConfig(level=logging.DEBUG)
import json

def test_prediction():
    '''

    :return:
    '''

    # Arrange
    payload = {'segment': {0: '00886add',  1: '00886add',  2: '00886add',
                            3: '00886add',  4: '00886add',  5: '00886add',
                            6: '00886add',  7: '00886add'},
                'sample_time': {0: '2009-02-04 04:44:12',  1: '2009-02-04 04:44:14',  2: '2009-02-04 04:44:20',
                                3: '2009-02-04 04:44:26',  4: '2009-02-04 04:44:32',  5: '2009-02-04 04:44:38',
                                6: '2009-02-04 04:44:44',  7: '2009-02-04 04:44:50'},
                'latitude': {0: 39.9853283,  1: 39.9853433,  2: 39.9854166,  3: 39.9854749,
                             4: 39.9855449,  5: 39.9855666,  6: 39.9855849,  7: 39.9856216},
                'longitude': {0: 116.3389483,  1: 116.3391383,  2: 116.3404399,
                              3: 116.3419583,  4: 116.3434716,  5: 116.3450066,
                              6: 116.3465516,  7: 116.3481583}}
    r = requests.post('http://localhost:8080/mode_prediction', json={'gps_trajectories': payload})

    assert (json.loads(r.content)[0] == 'airplane')

