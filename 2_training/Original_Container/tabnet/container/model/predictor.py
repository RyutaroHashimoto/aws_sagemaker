from __future__ import print_function

import json
import os
from io import StringIO

import flask
import pandas as pd
import torch
from model import TabNet_Regressor as model
from pytorch_tabnet.tab_model import TabNetRegressor as original

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = model()
            cls.model.model = original()
            cls.model.load_model(os.path.join(model_path, 'model.pt'))
            if torch.cuda.is_available():
                cls.model.model.device = 'cuda'
            else:
                cls.model.model.device = 'cpu'

        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""

        clf = cls.get_model()

        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)
    elif flask.request.content_type == 'application/json':
        res = flask.request.data.decode('utf-8')
        data = pd.read_json(json.loads(res), orient='split')
        # return flask.Response(response=res, status=415, mimetype='application/json')
    else:
        res = {
            "Error":
                'This predictor only supports CSV data or json'
        }
        return flask.Response(response=json.dumps(res), status=415, mimetype='application/json')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)

    if flask.request.content_type == 'text/csv':
        # Convert from numpy to CSV
        out = StringIO()
        pd.DataFrame(predictions).to_csv(out, header=False, index=False)
        response = out.getvalue()
        return flask.Response(response=response, status=200, mimetype='text/csv')
    elif flask.request.content_type == 'application/json':
        # Convert from numpy to JSON
        response = json.dumps({
            "predictions": predictions.tolist()
        })
        return flask.Response(response=response, status=200, mimetype='application/json')
