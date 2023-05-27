#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importación librerías
import pandas as pd
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import matplotlib.pyplot as plt 
#import seaborn as sns
import re
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

#from keras.layers import Activation
#from livelossplot import PlotLossesKeras
#from keras import backend as K
#import keras.optimizers as opts
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#from keras.callbacks import EarlyStopping, ModelCheckpoint

import joblib
# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields
import os
from predict_genre_movies import predict_genre
os.chdir('..')


# In[ ]:


# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Movie genre classification API',
    description='Movie genre classification API')

ns = api.namespace('predict', 
     description='MODELO A USAR')

# Definición argumentos o parámetros de la API
parser = api.parser()
# Argumento PLOT
parser.add_argument('Plot', type=str, required=True, help='Plot of the movie', location='args')
# Argumento mileage
#parser.add_argument('Mileage', type=int, required=True, help='Driven mileages', location='args')
# Argumento State
#parser.add_argument('State', type=str, required=True, help='State', location='args')
# Argumento State
#parser.add_argument('Make', type=str, required=True, help='Make', location='args')
# Argumento Model
#parser.add_argument('Model', type=str, required=True, help='Model', location='args')

resource_fields = api.model('Resource', {
    'result': fields.String(description='Result of the prediction')
})

@ns.route('/')
class Movie_prediction_Api(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        #retorno=predict_genre(args['Plot'])
        return {"result": predict_genre(args['Plot'])}, 200
# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
