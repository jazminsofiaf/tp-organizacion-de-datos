import pandas as pd
import numpy as np

#faltan
# 	created_on
#place_name
#floor 	rooms

#ya estan
#lat 	lon
#property_type 	state_name
#surface_total_in_m2 	surface_covered_in_m2
#expenses 	price_aprox_usd 	price_usd_per_m2
#description

def standarize(column):
    return (column - column.mean()) / column.std()

def one_hot_encode(column):
    return np.eye(column.max()+1)[column.reshape(-1)]

def coordenates_encode(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon),
    z = np.sin(lat)
    xy = np.append(x, y, axis=1)
    return np.append(xy, z, axis=1)

def unite_subcols(cols1, cols2):
    return np.append(cols1,cols2,axis=1)

dataset = pd.read_csv("../data_filled_ready_to_train.csv")
prices = standarize(dataset[["price_aprox_usd"]].values))
expenses = standarize(dataset[["expenses"]].values))
surfacetotal = standarize(dataset[["surface_total_in_m2"]].values))
surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values))
proptypes = one_hot_encode(dataset[["property_type"]].values - 1)
states = one_hot_encode(dataset[["state_name"]].values)
xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)



place_name = one_hot_encode(dataset[["place_name"]].values)

def normalize_dates(dates):
    max_date = dates.max()
    min_date = dates.min()
    min_norm = -1
    max_norm =1
    return (dates- min_date) *(max_norm - min_norm) / (max_date-min_date) + min_norm

created_on = normalize_dates(dataset[["created_on"]])
