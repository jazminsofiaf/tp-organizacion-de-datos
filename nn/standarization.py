import pandas as pd
import numpy as np

#faltan
# 	created_on
#floor 	rooms

#ya estan
#place_name
#lat 	lon
#property_type 	state_name
#surface_total_in_m2 	surface_covered_in_m2
#expenses 	price_aprox_usd 	price_usd_per_m2
#description

def standarize(column):
    return (column - column.mean()) / column.std()

def one_hot_encode(column, encodesize=None):
    if encodesize:
        return np.eye(encodesize)[column.reshape(-1)]
    else:
        return np.eye(column.max()+1)[column.reshape(-1)]

def coordenates_encode(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    xy = np.append(x, y, axis=1)
    return np.append(xy, z, axis=1)

def unite_subcols(cols1, cols2):
    return np.append(cols1,cols2,axis=1)

def run_on_csv():
    dataset = pd.read_csv("../data_filled_ready_to_train.csv")
    prices = standarize(dataset[["price_aprox_usd"]].values)
    expenses = standarize(dataset[["expenses"]].values)
    surfacetotal = standarize(dataset[["surface_total_in_m2"]].values)
    surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(dataset[["property_type"]].values - 1)
    states = one_hot_encode(dataset[["state_name"]].values)
    xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)
