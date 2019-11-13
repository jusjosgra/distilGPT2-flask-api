import os
from flask import Flask, jsonify, request, make_response
from flask_restplus import Api, Resource, fields

from distilgpt2_model import predict

#HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}

flask_app = Flask(__name__)
app = Api(app = flask_app, 
        version = "1.0", 
        title = "distilGPT2 text generation", 
        description = "Generate text continuation given a prompt string.")

name_space = app.namespace('predict_text', description='Prediction APIs')

model = app.model('Prediction params', 
                {'textField': fields.String(required = True, 
                                            description="Text Field", 
                                            help="Text Field cannot be blank")
                }
                )

@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        try:
            formData = request.json
            data = [val for val in formData.values()]
            prediction = predict(data[0])
            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": "Prediction: " + str(prediction)
                })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })
