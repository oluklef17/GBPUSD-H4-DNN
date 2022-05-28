import pickle
import flask
from flask import Flask, render_template, request
import requests
import numpy as np

url = requests.get('https://firebasestorage.googleapis.com/v0/b/mt5-remote-trade-copier.appspot.com/o/trades%2FFOREX%20AI%20DATA%2FMODELS%2FGBPUSD_model.pkl?alt=media&token=8a692b84-6809-41a7-b710-f0d9339195bd')

with open('model.pkl', 'wb') as f:
  f.write(url.content)
  
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/<string:data>', methods=['POST', 'GET'])
def predict(data):
  parts = data.split(',')
  d1 = []
  d2 = []
  dataset = []
  
  for i in parts:
    d1.append(float(i))
    d2.append(float(i))
  
  dataset = np.array([d1,d2])
  
  result = model.predict(dataset)
  return str(result)

if __name__ == '__main__':
 app.run(host='127.0.0.1', port=8080, debug=False)
