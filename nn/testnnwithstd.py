import numpy as np
import pandas as pd
from nn import *
from standarization import *


def get_nn_parameters(epochs=5, trainsize=0.8):
    dataset = pd.read_csv("../data_filled_ready_to_train.csv")

    prices = dataset[["price_aprox_usd"]].values
    pricesstd = standarize(prices)
    expenses = standarize(dataset[["expenses"]].values)
    surfacetotal = standarize(dataset[["surface_total_in_m2"]].values)
    surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(dataset[["property_type"]].values - 1)
    states = one_hot_encode(dataset[["state_name"]].values)
    xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)

    datastd = unite_subcols(expenses, surfacetotal)
    datastd = unite_subcols(datastd, surfacecovered)
    datastd = unite_subcols(datastd, proptypes)
    datastd = unite_subcols(datastd, states)
    datastd = unite_subcols(datastd, xyz)

    parameters = None
    for value in range(1,epochs):
        msk = np.random.rand(len(datastd)) < trainsize
        train = datastd[msk]
        test = datastd[~msk]
        trainprices = pricesstd[msk]
        testprices = prices[~msk]

        sh = testprices.shape[0]
        testprices = testprices.reshape((1,sh))
        sh = trainprices.shape[0]
        trainprices = trainprices.reshape((1,sh))

        parameters = nn_model(train.T,trainprices,10,num_iterations=5000,parameters=parameters)
    print("params found")
    predictions = predict(parameters, test.T, testprices.std(), testprices.mean())

    #predictions = predictions.T * testprices.std() + testprices.mean()

    print("predict: ", predictions.T)
    print("real: ",testprices.T)
    mse = np.square(testprices.T - predictions).mean()
    print(mse)
    return parameters, prices.std(), prices.mean()

    #test_hidden_layers(train.T,trainprices, test.T, testprices.T)

def predict_test(parameters, pricesstd, pricesmean):
    dataset = pd.read_csv("../data_filled_ready_to_predict.csv")
    ids = dataset[["id"]]

    expenses = standarize(dataset[["expenses"]].values)
    surfacetotal = standarize(dataset[["surface_total_in_m2"]].values)
    surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(dataset[["property_type"]].values - 1)
    states = one_hot_encode(dataset[["state_name"]].values)
    xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)

    datastd = unite_subcols(expenses, surfacetotal)
    datastd = unite_subcols(datastd, surfacecovered)
    datastd = unite_subcols(datastd, proptypes)
    datastd = unite_subcols(datastd, states)
    datastd = unite_subcols(datastd, xyz)

    predictions = predict(parameters, datastd.T, pricesstd, pricesmean)
    results = unite_subcols(ids, predictions.T)
    np.savetxt("submit.csv", results, fmt='%i', delimiter=",")

if __name__ == '__main__':
    parameters,pricestd,pricemean = get_nn_parameters()
    predict_test(parameters,pricestd,pricemean)
