from flask import Flask, jsonify
from flask_cors import CORS

import requests

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/get_suggestion')
def get_suggestion():
    url = 'https://virtual-tryon-backend-974u.onrender.com/'
    url = f'{url}/generate_looks_from_top'
    data = {
        "top_id": "4865",
    }
    response = requests.post(url, json=data)
    return jsonify(response.text)

if __name__ == '__main__':
    app.run()

