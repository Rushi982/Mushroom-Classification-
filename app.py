# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:24:39 2023

@author: DELL
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle as pkl
#from gevent.pywsgi import WSGIServer


app=Flask(__name__)

model=pkl.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    m=round(prediction[0],2)
    if m==0:
        g = "Edible"
    else:
        g="Poisonous"
    return render_template('index.html', prediction_text = "The Mushroom is {}".format(g))

if __name__=='__main__':
    app.run(debug=False)
   #http_server = WSGIServer(("localhost",8080), app)
  # http_server.serve_forever()
   
   #import socket
   #socket.getaddrinfo("localhost", 8080)