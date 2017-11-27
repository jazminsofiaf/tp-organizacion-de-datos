import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


def linear_regression(df_to_predict, X_train, y_train):
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_price_usd_predicted = regr.predict(df_to_predict)
    return y_price_usd_predicted

df_to_predict= pd.read_csv('../tp2/data_filled_ready_to_predict2.csv',encoding='UTF-8')
df= pd.read_csv('../tp2/data_filled_ready_to_train2.csv',encoding='UTF-8')

predictions_total = pd.DataFrame()
places = df_to_predict.place_name.append(df.place_name)
places = places.drop_duplicates(keep = "first")

for place in places:
    
    print place
    X = df_to_predict.loc[df_to_predict.place_name == place,:]
    if(X.shape[0] == 0):
        continue 
    
    x_train = df.loc[df.place_name == place,:]
    if(x_train.shape[0] == 0):
        print "no existe barrio para entrenar",place
        x_train = df
    
    y_train = x_train.price_aprox_usd
    
    
    
    columns = ['created_on','property_type','lat','lon', 
                 'surface_total_in_m2','surface_covered_in_m2',
                 'floor','rooms','expenses']
    
    X['price_usd']  = linear_regression(X.loc[:,columns] , x_train.loc[:,columns] , y_train)
    
    ready_df= X.loc[:,['id','price_usd']]
    
    predictions_total = pd.concat([predictions_total,ready_df])
    
    
predictions_total.to_csv("prediccion1.csv",encoding='utf-8',index=False)