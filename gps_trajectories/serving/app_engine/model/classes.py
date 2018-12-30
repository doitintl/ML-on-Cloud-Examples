
import numpy as np


class GPSClasses:

    def __init__(self):

        # TODO: initialize this with the saved LabelEncoder from GCS
        self.classes = ['airplane', 'bike', 'boat', 'bus', 'car', 'motorcycle', 'run',
                           'subway', 'taxi', 'train', 'walk']

    @staticmethod
    def parse_results(probs, classes):
        """
        Converts a list of class probabilities to the name of the most probable class
        :param probs: list of probabilities list
        :param classes: map between index to class name
        :return:
        """

        return [classes[np.argmax(np.array(x))] for x in probs['predictions']]
