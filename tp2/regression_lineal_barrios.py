import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df_to_predict= pd.read_csv('../tp1/data_filled_ready_to_predict.csv',encoding='UTF-8')
df= pd.read_csv('../tp1/data_filled_ready_to_train.csv',encoding='UTF-8')


def linear_regression(df_to_predict, X_train, y_train):
    scaler = StandardScaler()
    x_train_standarized =scaler.fit_transform(x_train.loc[:,columns])
    X_standarized = scaler.fit_transform(X.loc[:,columns])
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_price_usd_predicted = regr.predict(df_to_predict)
    return y_price_usd_predicted

predictions_total = pd.DataFrame()
places = df_to_predict.place_name.drop_duplicates(keep='first').sort_values()
for place in places:
   
    x_train = df.loc[df.place_name == place,:]
    y_train = x_train.price_aprox_usd
    X = df_to_predict.loc[df_to_predict.place_name == place,:]
    
    columns = ['created_on','property_type','lat','lon', 
                 'surface_total_in_m2','surface_covered_in_m2',
                 'floor','rooms','expenses']
    
    X['price_usd']  = linear_regression(X.loc[:,columns] , x_train.loc[:,columns] , y_train)
    
    ready_df= X.loc[:,['id','price_usd']]
    
    predictions_total = pd.concat([predictions_total,ready_df])
    
    
predictions_total.to_csv("prediccion3.csv",encoding='utf-8',index=False)