# -*- coding: utf-8 -*-
"""
@author: hzumaeta
"""
#import traceback
from flasgger import Swagger
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)
swagger = Swagger(app)

# inputs
model_directory = 'model'
model_file_name = '%s/lmmodel.pkl' % model_directory
model_columns_file_name = '%s/lmmodel_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
clf = None

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

def predict(json_ask):
    json_ = json_ask
    query = pd.get_dummies(pd.DataFrame(json_))
    query = query.reindex(columns=model_columns, fill_value=0)
    prediction = clf.predict(query).tolist()
    predict_prob = clf.predict_proba(query)	.tolist()
    prediction_final = {"prediction": prediction,"prob":predict_prob}
    return (jsonify(prediction_final))

@app.route("/predict", methods=['POST'])
def post():
    """
    Predict whether commodity will be a loss for client

    ---
    tags:
      - losstransaction predictions
    parameters:
      - in: body
        name: body
        schema:
          id: User
          required:
            - customertier
            - revenue
            - shipstate
            - storestate
          properties:
            customertier:
              type: integer
              description: customer
              default: 5
            timesreturned:
              type: integer
              description: how many times this product type from this customer was returned
              default: 2
            shipstate:
              type: string
              description: where commodity shipped from
              default: CA
            storestate:
              type: string
              description: store location state
              default: WA

    responses:
        200:
            description: A single recommendation item
            schema:
              id: rec_response
              properties:
                prediction:
                  type: integer
                  description: The id of the opening
                  default: 0
        204:
             description: No recommendation or no model loaded
    """

    return (predict(request.json))

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    try:
        clf = joblib.load(model_file_name)
        print ('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print ('model columns loaded')

    except Exception as e:
        print ('No model here')
        print ('Train first')
        print (str(e))
        clf = None

    app.run(port=800)
