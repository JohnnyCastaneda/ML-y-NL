#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importación librerías
import pandas as pd
import numpy as np
#from random import randrange
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
#from m09_model_deployment import predict_price

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib

#import os
#os.chdir('..')


# In[ ]:


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Pricing prediction for used cars API',
    description='Pricing prediction for used cars API')

ns = api.namespace('predict', 
     description='Pricing XGBoost')

# Definición argumentos o parámetros de la API
parser = api.parser()
# Argumento año
parser.add_argument(
    'Year', 
    type=int, 
    required=True, 
    help='Year of manufacturation', 
    location='args')
# Argumento mileage
parser.add_argument(
    'Mileage', 
    type=int, 
    required=True, 
    help='Driven mileages', 
    location='args')
# Argumento State
parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='State', 
    location='args')
# Argumento State
parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Make', 
    location='args')
# Argumento Model
parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Model', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class PricingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_price(args['Year'],args['Mileage'],args['State'],args['Make'],args['Model'])
        }, 200
    
# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

