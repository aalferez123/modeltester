from google.cloud import aiplatform
from flask import Flask, render_template, request, jsonify
from typing import Dict, List, Union
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import os

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    # for prediction in predictions:
    #     print(" prediction:", prediction)

    return predictions

# Overides parameters for inferences.
# If you encounter the issue like `ServiceUnavailable: 503 Took too long to respond when processing`,
# you can reduce the max length, such as set max_tokens as 20.

############### Application ########################

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_model():
    project_id = request.form['project_id']
    endpoint_id = request.form['endpoint_id']
    query_text = request.form['query']
    temperature = request.form ['temperature']
    max_tokens = request.form['max_tokens']
    location = request.form['location']  # Assuming you get location from the form
    api_endpoint = location+"-aiplatform.googleapis.com"

    # Construct the 'instances' list (adjust as needed for your model)
    instances = [
        {
            "prompt": "<|start_header_id|>user<|end_header_id|>"+ query_text +"<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "max_tokens": 1000,  # You can customize these parameters
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1.0,
            "raw_response": True
        }
    ]


    try:
        response = predict_custom_trained_model_sample(
            project=project_id,
            endpoint_id=endpoint_id,
            instances=instances
        )
        for responses in response:
            print(responses)
        #print(response)
        # Extract the prediction 
        # prediction = response.predictions[0]
        # for prediction in prediction:
        #  print(" prediction:", prediction)
        # return jsonify({"model_response": response})
        return responses
     
    except Exception as e: # Log the full traceback
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')