import pandas as pd
import numpy as np
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def linear_regression(df_to_predict, X_train, y_train):
    scaler = StandardScaler()
    x_train_standarized =scaler.fit_transform(X_train)
    X_standarized = scaler.fit_transform(df_to_predict)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_price_usd_predicted = regr.predict(df_to_predict)
    return y_price_usd_predicted

def knn(x_train, y_train, x_to_predict):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train,  y_train) 
    return neigh.predict(x_to_predict)



df= pd.read_csv('../tp1/data_filled_ready_to_train.csv',encoding='UTF-8')
df_to_predict= pd.read_csv('../tp1/data_filled_ready_to_predict.csv',encoding='UTF-8')

#divido el dataframe en bins segun el precio por metro cuadrado de la propiedad
#el menor valor esta al rededor de 200
df['bin'] = df.price_usd_per_m2.map(lambda value: int((value-200)/50) )

#predigo por knn (geografico) en que bin se encontrara
df_to_predict['bin'] = knn(df.loc[:,['lat','lon']], df['bin'],df_to_predict.loc[:,['lat','lon']])

bins = df_to_predict.place_name.drop_duplicates(keep='first')

prediccion_total = pd.DataFrame()
for b in bins:
    
    df_bin = df_to_predict.loc[df_to_predict.bin == b,:]
    x = df.loc[df.bin == b,:]
    if(df_bin.shape[0] == 0 ):
        continue
    columns = ['created_on','property_type','lat','lon', 'place_name','state_name',
                 'surface_total_in_m2','surface_covered_in_m2','description',
                 'floor','rooms','expenses']
    
    df_bin['price_usd']  = linear_regression(df_bin.loc[:,columns], x.loc[:,columns], x.price_aprox_usd)
    
    
    ready_df= df_bin.loc[:,['id','price_usd']]
    prediccion_total = pd.concat([prediccion_total,ready_df])

    
prediccion_total.price_usd = prediccion_total.price_usd.map(lambda v: int(v))
prediccion_total.to_csv("prediccion4.csv",encoding='utf-8',index=False)
