import os
from flask import Flask, jsonify, request

import IPython
from distilgpt2_test import predict
from urllib.parse import parse_qs


HEADERS = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'text/plain'}

def flask_app():
    app = Flask(__name__)


    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up'

    @app.route('/predict_text', methods=['POST'])
    def start():
        text = request.form['text']
        pred = predict(text)
        return jsonify({"Predicted text: ":pred})
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')