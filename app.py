'''
Flask API to make predictions
'''
import logging
import os
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request, send_file
import base64
import predict
import time
import json
import cv2
import numpy as np
import io
from google.cloud import storage

app = Flask(__name__)
cors = CORS(app)


@app.route('/predict', methods=['POST'])
def get_predictions():
    '''Function to call when a POST request is made.

        Parameters:
            None
        Return Value:
            Predictions List.
    '''

    if request.method == 'POST':
        image_data = request.files['file']
        image_data.save("image.png")
        predict.get_lungs()
        storage_client = storage.Client.from_service_account_json('gcs.json')
        bucket = storage_client.bucket('pathology-bucket')
        blob = bucket.blob('chest_xray.png')
        blob.upload_from_filename('chest_xray.png')

    return send_file('chest_xray.png', mimetype='image/png', as_attachment=True)


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
