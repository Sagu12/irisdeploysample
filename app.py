# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:53:33 2021

@author: Sagnik.Banerjee
"""

from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server


import joblib


model= joblib.load(r"iris.pkl")
app= Flask(__name__)

def predict():
    sepallength= input("Enter the sepal length: ", type=FLOAT)
    sepalwidth= input("Enter the sepal width: ", type=FLOAT)
    petallength= input("Enter the petal length: ", type=FLOAT)
    petalwidth= input("Enter the petal width: ", type=FLOAT)
    
    
    prediction= model.predict([[sepallength, sepalwidth, petallength, petalwidth]])
    
    
    if prediction == 0:
        put_text("The prediction is setosa")
        
    elif prediction == 1:
        put_text("The prediction is versicolor")
        
    else:
        put_text("The prediction is virginica")
        
app.add_url_rule('/iris', 'webio_view', webio_view(predict), methods= ["GET", "POST", "OPTIONS"])


#deployment to heroku use this
if __name__== "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args= parser.parse_args()
    
    start_server(predict, port= args.port)





#this is used for running in local server or local host
#app.run(host='localhost', port=80, debug=True)
        
    
