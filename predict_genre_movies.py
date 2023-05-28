#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#
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
os.chdir('..')


# In[ ]:


def predict_genre(plot):
    dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
    ######## Limpieza de texto ###############
    def lemmatize_words(plot):
        lem = WordNetLemmatizer()
        lemmatized_words = [lem.lemmatize(word, 'v') for word in plot.split()]
        lemmatized_sentence=' '.join(lemmatized_words)
        return lemmatized_sentence

    def clean_text(plot):
        # remove backslash-apostrophe 
        text = re.sub("\'", "", plot) 
        # remove everything except alphabets 
        text = re.sub("[^a-zA-Z]"," ",plot) 
        # remove whitespaces 
        text = ' '.join(plot.split()) 
        # convert text to lowercase 
        text = plot.lower()    
        return text
    
    stop_words = set(stopwords.words('english'))
    
    def remove_stopwords(plot):
        no_stopword_text = [w for w in plot.split() if not w in stop_words]
        return ' '.join(no_stopword_text)
    plot=lemmatize_words(plot)
    plot=clean_text(plot)
    plot=remove_stopwords(plot)
    
    ###### Transformación de texto ##########    
    vect = joblib.load('/home/ubuntu/ML-y-NL/vect.pkl')
    X_dtm = vect.transform([plot])
    X_dtm=X_dtm.toarray()
    
    ##### Carga del modelo y generación de predicción ####
    MODELO = joblib.load('/home/ubuntu/ML-y-NL/clf.pkl')
    prediccion=MODELO.predict_proba(X_dtm)
    
    ##### Dar formato a la predicción ##### 
    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
    res = pd.DataFrame(prediccion, columns=cols)
    genero=res.idxmax(axis=1)[0]
    probabilidad=res.max(axis=1)[0]
    retorno="El género más probable es: "+str(genero)+" con una probabilidad de: "+str(probabilidad)
        # Obtén la primera fila del DataFrame
    fila = res.iloc[0]
    b=''
    # Imprime los nombres de las columnas y los valores correspondientes
    for columna, valor in fila.iteritems():
        b=b+' || '+(f'{columna}: {valor}')    
    
    return retorno+", la probabilidad de todos los géneros: "+b
    


