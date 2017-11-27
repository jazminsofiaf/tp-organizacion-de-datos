import numpy as np
import pandas as pd
from nn import *
from standarization import *

def test_states(epochs=5, trainsize=0.8, learning_rate=None, nh=10):
    dataset = pd.read_csv("../data_filled_ready_to_train.csv")

    prices = dataset[["price_aprox_usd"]].values
    pricesstd = standarize(prices)

    states = one_hot_encode(dataset[["state_name"]].values, encodesize=5)
    places = one_hot_encode(dataset[["place_name"]].values, encodesize=542)
    expenses = standarize(dataset[["expenses"]].values)
    surfacetotal = standarize(dataset[["surface_total_in_m2"]].values)
    surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(dataset[["property_type"]].values - 1, encodesize=6)
    xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)
    descriptions = standarize(dataset[["description"]].values)

    datastd = unite_subcols(expenses, surfacetotal)
    datastd = unite_subcols(datastd, surfacecovered)
    datastd = unite_subcols(datastd, proptypes)
    datastd = unite_subcols(datastd, places)
    datastd = unite_subcols(datastd, states)
    datastd = unite_subcols(datastd, xyz)
    datastd = unite_subcols(datastd, descriptions)

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

        parameters = nn_model(train.T, trainprices, nh,num_iterations=300,
                            parameters=parameters, learning_rate=learning_rate)

        predictions = predict(parameters, test.T, testprices.std(), testprices.mean())
        mse = np.square(testprices.T - predictions).mean()
        print("MSE: %s" % mse)
    #predict_test(parameters, prices.std(), prices.mean(),filename='submitlayers{}.csv'.format(nh))


def predict_test(parameters, pricesstd, pricesmean, filename="submit.csv"):
    """
    For predicting de test set, saving a file for submition.
    Args:
    -parameters: result from training the NN
    -pricesstd: standard deviation of prices from training
    -pricesmean: mean of prices from training set
    """
    dataset = pd.read_csv("../data_filled_ready_to_predict.csv")
    ids = dataset[["id"]]

    states = one_hot_encode(dataset[["state_name"]].values, encodesize=5)
    places = one_hot_encode(dataset[["place_name"]].values, encodesize=542)
    expenses = standarize(dataset[["expenses"]].values)
    surfacetotal = standarize(dataset[["surface_total_in_m2"]].values)
    surfacecovered = standarize(dataset[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(dataset[["property_type"]].values - 1, encodesize=6)
    xyz = coordenates_encode(dataset[["lat"]].values,dataset[["lon"]].values)
    descriptions = standarize(dataset[["description"]].values)

    datastd = unite_subcols(expenses, surfacetotal)
    datastd = unite_subcols(datastd, surfacecovered)
    datastd = unite_subcols(datastd, proptypes)
    datastd = unite_subcols(datastd, places)
    datastd = unite_subcols(datastd, states)
    datastd = unite_subcols(datastd, xyz)
    datastd = unite_subcols(datastd, descriptions)

    predictions = predict(parameters, datastd.T, pricesstd, pricesmean)
    results = unite_subcols(ids, predictions.T)
    np.savetxt(filename, results, fmt='%i', delimiter=",")

if __name__ == '__main__':
    for i in [10,20,50,100]:
        print("hidden layer: %i" % i)
        test_states(learning_rate=.5, epochs=3,nh=i)
