import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import utils

#Función a usar para predecir
def linear_regression(df_to_predict, x_train, y_train):
    scaler = StandardScaler()
    x_train_standarized =scaler.fit_transform(x_train)
    X_standarized = scaler.fit_transform(df_to_predict)
    regr = LinearRegression()
    regr.fit(x_train_standarized,y_train)
    y_price_usd_predicted = regr.predict(X_standarized)
    return y_price_usd_predicted

train_set = pd.read_csv('data_filled_ready_to_train.csv')
train_set = train_set.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
       'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
       'rooms', 'expenses','price_aprox_usd']]

#Saco las ultimas dos filas por un tema de división del dataset en partes para el entrenamiento
train_set.drop(train_set.index[78656], inplace=True)
train_set.drop(train_set.index[78655], inplace=True)

df_to_predict = pd.read_csv('data_filled_ready_to_predict.csv')
predic_set = df_to_predict.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]

train_set_aux = train_set.copy()

def setear_en_cero(row): return 0
train_set_aux['knn'] = train_set_aux.apply(lambda row: setear_en_cero(row),axis=1)

puntos_division = len(train_set) // 5

z=0
j = puntos_division
total_pred = []
for x in range(5):
    train_set_temporal = train_set[z:j]
    train_set_temporal = train_set_temporal.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]
    frames = [train_set[:z], train_set[j:]]
    X_train = pd.concat(frames)
    X_train = X_train.astype(int)
    Y_train = X_train.price_aprox_usd
    X_train = X_train.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(X_train, Y_train)
    prediccion = neigh.predict(train_set_temporal)
    prediccion = prediccion.tolist()
    total_pred += prediccion
train_set_aux['knn'] = total_pred


#Clasifico con knn a los datos a predecir
predic_set = df_to_predict.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]
X_train = train_set[:]
X_train = X_train.astype(int)
Y_train = X_train.price_aprox_usd
X_train = X_train.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses',]]
neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X_train, Y_train)
prediccion = neigh.predict(predic_set)
prediccion = prediccion.tolist()

predic_set['knn'] = prediccion


#Hago K-means para el dataset de entrenamiento


# Object KMeans
kmeans = sklearn.cluster.KMeans(n_clusters=6, init="k-means++", n_init=10, max_iter=500, tol=0.0001,
                        precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=1)#, algorithm="auto")

# Calculate Kmeans
kmeans.fit(train_set_aux)

dic_clusters = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

#Hago K-means para el dataset de validación

predict_set_aux = predic_set

import numpy as np
def setear_en_cero(row): return 0
#Columna que luego será llenada con las predicciones
predict_set_aux['prediccion'] = predict_set_aux.apply(lambda row: setear_en_cero(row),axis=1)

# Object KMeans
kmeans2 = sklearn.cluster.KMeans(n_clusters=6, init="k-means++", n_init=10, max_iter=500, tol=0.0001,
                        precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=1)#, algorithm="auto")

# Calculate Kmeans
kmeans2.fit(predict_set_aux)

dic_clusters_predict = {i: np.where(kmeans2.labels_ == i)[0] for i in range(kmeans2.n_clusters)}

#Una vez que se los clusters de cada dato extraigo del dataset de entrenamiento los datos del cluster 0 y los uso para predecir
#los datos del dataset de predicción pertenecientes también al cluster 0. Luego repito procedimiento para cada cluster.
for clave in dic_clusters:

    #Creación de datasets donde se almacenarán los datos de cada dataset pertenecientes a determinado cluster (según la iteración)
    df = pd.DataFrame(columns=['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses','price_aprox_usd'])
    df2 = pd.DataFrame(columns=['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses'])

    for j in range(0,len(dic_clusters[clave])):
        df.loc[j] = train_set_aux.loc[dic_clusters[clave][j]]


    for k in range(0,len(dic_clusters_predict[clave])):
        df2.loc[k] = predict_set_aux.loc[dic_clusters_predict[clave][k]]

    df_to_predict = df2.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]

    X_train = df
    Y_train = X_train.price_aprox_usd
    X_train = X_train.loc[:,['created_on', 'property_type', 'lat', 'lon', 'place_name', 'state_name',
               'surface_total_in_m2', 'surface_covered_in_m2', 'description', 'floor',
               'rooms', 'expenses']]

    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rf.fit(X_train, Y_train)
    total_predicciones = rf.predict(df_to_predict)

    i = 0
    for punto in dic_clusters_predict[clave]:
        if i < len(total_predicciones):
            predict_set_aux.loc[punto, 'prediccion'] = total_predicciones[i]
        i += 1
#Reabro el csv para obtener los ids
predict_set = pd.read_csv('data_filled_ready_to_predict.csv')
ids = predict_set.id
predicciones = predict_set_aux.prediccion
resultado = pd.DataFrame({'id': ids, 'price_usd' : predicciones})
resultado.to_csv("prediccion2.csv",encoding='utf-8',index=False)
