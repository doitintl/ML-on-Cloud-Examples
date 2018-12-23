from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from flask import Flask, request
import json
from model.feature_extraction import GPSTrajectoriesModel

VERSION_NAME = 'v2'
MODEL_NAME = 'transportation_mode'
PROJECT_ID = 'gad-playground-212407'

app = Flask(__name__)

# Initialize objects
gps_model = GPSTrajectoriesModel()


@app.route('/mode_prediction', methods=['POST'])
def generate_predictions():
    """
    Request handler
    :return:
    """
    payload = request.json
    records = payload['gps_trajectories']

    # Extract features from data
    features = GPSTrajectoriesModel.extract_features(records)

    return json.dumps(_query_model(features))


def _query_model(data):
    """
    Uses a Cloud Machine Learning Engine client to generate predictions
    :param features: tuple of strings with account-billing-ids
    :return: dictionary of predictions with form: {"account_billing_id" : prediction, ...}
    """
    model_name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
    model_name += '/versions/{}'.format(VERSION_NAME)

    credentials = GoogleCredentials.get_application_default()

    ml = discovery.build('ml', 'v1', credentials=credentials)

    # Create a dictionary with the fields from the request body.
    request_body = {"instances": data}

    # Create a request to call projects.models.create.
    request = ml.projects().predict(
        name=model_name,
        body=request_body)
    response = request.execute()

    response = GPSTrajectoriesModel.parse_results(response, gps_model.classes)

    return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
