#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Importación librerías
#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from random import randrange

from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

def predict_price(year,mileage,state,make,model):
    dummie_state=str("State_")+state
    dummie_make=str("Make_")+make
    MODELO = joblib.load(os.getcwd()+'/Modelo_prices.pkl') 
    dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
    # eliminar espacios en la columna "Make"
    dataTraining["Make"] = dataTraining["Make"].apply(lambda x: x.strip())

    # eliminar espacios en la columna "State"
    dataTraining["State"] = dataTraining["State"].apply(lambda x: x.strip())
    # Definir el número de particiones en el precio
    k=60
    ############################## Dummies para Marca ####################################

    dummies = pd.get_dummies(dataTraining['Make'], prefix='Make')
    dummies=dummies.drop('Make_Freightliner', axis=1)
    dataTraining = pd.concat([dataTraining, dummies], axis=1)

    ######################## Dummies para los modelos ##########################

    # Crear un diccionario vacío para almacenar los resultados
    promedios_por_marca = {}

    # recorrer los valores distintos de la columna "Make"
    for marca in dataTraining["Model"].unique():
        # obtener el promedio de la columna "Price" para los registros donde "Make" es igual a la marca actual
        promedio = dataTraining.loc[dataTraining["Model"] == marca, "Price"].mean()
        # agregar la marca y su promedio al diccionario
        promedios_por_marca[marca] = promedio
    diccionario_ordenado = dict(sorted(promedios_por_marca.items(), key=lambda x: x[1], reverse=True))


    # Creamos los nombres de las particiones
    column_names = []
    for i in range(0, 60000, int(60000/k)):
        name = f"{i}-{i+int(60000/k)}"
        column_names.append(name)
    # Creamos las columnas con el nombre de la partición en el dataframe

    for i in range(len(column_names)):
        dataTraining[column_names[i]]=0

    # Creamos diccionario con el rango de precios como llaves y los modelos que se encuentran en ese rango como valores
    rango_precios = {}

    for column in column_names:
        start, end = column.split("-")
        start, end = int(start), int(end)
        rango_precios[column] = [key for key, value in diccionario_ordenado.items() if start <= value <= end]

    ############################## Dummies para State ####################################
    dummies = pd.get_dummies(dataTraining['State'], prefix='State')
    dataTraining = pd.concat([dataTraining, dummies], axis=1)

    ############################# Borramos columnas ya procesadas #######################
    dataTraining = dataTraining.drop(['State', 'Make', 'Model',"Price"], axis=1)

    #################### Creación y llenado del dataframe para predecir ################
    df = pd.DataFrame(np.zeros((1, len(dataTraining.columns))), columns=dataTraining.columns)
    df = df.astype(int)
    df.iloc[0] = df.iloc[0].round(1)
    df.loc[0, 'Year'] = year
    df.loc[0, 'Mileage'] = mileage
    df.loc[0, dummie_state] = 1
    df.loc[0, dummie_make] = 1
    for key, value in rango_precios.items():
        if model in value:
            rango = key
            break
    df.loc[0, rango] = 1

    prediccion=int(MODELO.predict(df))
    return prediccion


# In[21]:


#predict_price(2017,9913,"TN","Audi","Wrangler")


# In[ ]:




