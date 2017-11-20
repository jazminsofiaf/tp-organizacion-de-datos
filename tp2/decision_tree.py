import pandas as pd
import numpy as np
from pandas import Series
from sklearn import tree

df_to_predict= pd.read_csv('../tp1/data_filled_ready_to_predict.csv',encoding='UTF-8')
df= pd.read_csv('../tp1/data_filled_ready_to_train.csv',encoding='UTF-8')

def decision_tree_prediction(X_train, y_train,x_to_predict):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)
    return clf.predict(x_to_predict)

X_train = df.loc[:,['created_on','property_type','lat','lon', 'place_name',
       'state_name', 'surface_total_in_m2', 'surface_covered_in_m2',
       'description', 'floor', 'rooms', 'expenses']]
y_train = df.price_aprox_usd

x_test = df_to_predict.loc[:,['created_on','property_type','lat','lon', 'place_name',
       'state_name', 'surface_total_in_m2', 'surface_covered_in_m2',
       'description', 'floor', 'rooms', 'expenses']]

df_to_predict['price_usd'] = decision_tree_prediction(X_train, y_train, x_test)

prediccion5 = df_to_predict.loc[:,['id','price_usd']]

prediccion5 .to_csv("prediccion5.csv",encoding='utf-8',index=False)