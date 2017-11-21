import numpy as np
import pandas as pd
from nn import *
from standarization import *


def get_nn_parameters(epochs=5, trainsize=0.8, learning_rate=None):
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
    for value in range(0,epochs):
        msk = np.random.rand(len(datastd)) < trainsize
        train = datastd[msk]
        test = datastd[~msk]
        trainprices = pricesstd[msk]
        testprices = prices[~msk]

        sh = testprices.shape[0]
        testprices = testprices.reshape((1,sh))
        sh = trainprices.shape[0]
        trainprices = trainprices.reshape((1,sh))

        parameters = nn_model(train.T, trainprices, 10,num_iterations=5000,
                            parameters=parameters, learning_rate=learning_rate)

        predictions = predict(parameters, test.T, testprices.std(), testprices.mean())

        #predictions = predictions.T * testprices.std() + testprices.mean()

        #print("predict: ", predictions.T)
        #print("real: ",testprices.T)
        mse = np.square(testprices.T - predictions).mean()
        print("MSE: %s" % mse)
    return parameters, prices.std(), prices.mean()

    #test_hidden_layers(train.T,trainprices, test.T, testprices.T)

def test_learning_rate(lrmin=0.1,lrmax=1,lrstep=0.2):
    """
    Compute the NN for diferents learning_rates in a range
    Args:
    """
    for lrate in np.arange(lrmin,lrmax,lrstep):
        print("learning rate: %s" % lrate)
        get_nn_parameters(epochs=1, learning_rate=lrate)


def predict_test(parameters, pricesstd, pricesmean):
    """
    For predicting de test set, saving a file for submition.
    Args:
    -parameters: result from training the NN
    -pricesstd: standard deviation of prices from training
    -pricesmean: mean of prices from training set
    """
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

def basic_run():
    parameters,pricestd,pricemean = get_nn_parameters(learning_rate=0.4, epochs=3)
    predict_test(parameters,pricestd,pricemean)


if __name__ == '__main__':
    basic_run()
    #test_learning_rate(lrmin=0.4,lrmax=0.6,lrstep=0.05)
