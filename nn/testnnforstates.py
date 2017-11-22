import numpy as np
import pandas as pd
from nn import *
from standarization import *

def test_states(epochs=5, trainsize=0.8, learning_rate=None):
    dataset = pd.read_csv("../data_filled_ready_to_train.csv")

    states = dataset.groupby(["state_name"])

    results = {}
    for g in states.groups.keys():
        state = states.get_group(g)
        #prepare data of state
        prices = state[["price_aprox_usd"]].values
        pricesstd = standarize(prices)
        expenses = standarize(state[["expenses"]].values)
        surfacetotal = standarize(state[["surface_total_in_m2"]].values)
        surfacecovered = standarize(state[["surface_covered_in_m2"]].values)
        proptypes = one_hot_encode(state[["property_type"]].values - 1, encodesize=6)
        xyz = coordenates_encode(state[["lat"]].values,state[["lon"]].values)
        places = one_hot_encode(state[["place_name"]].values, encodesize=542)
        #descriptions = state[["description"]]

        datastd = unite_subcols(expenses, surfacetotal)
        datastd = unite_subcols(datastd, surfacecovered)
        datastd = unite_subcols(datastd, proptypes)
        datastd = unite_subcols(datastd, places)
        datastd = unite_subcols(datastd, xyz)
        #datastd = unite_subcols(datastd, descriptions)

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
            mse = np.square(testprices.T - predictions).mean()
            print("MSE: %s" % mse)
        predict_test(g, parameters, prices.std(), prices.mean(),filename=str(g)+'submit.csv')

        #results[g] = {'parameters': parameters, 'pricestd': prices.std(), 'pricemean': prices.mean()}

    return results

def predict_test(g, parameters, pricesstd, pricesmean, filename="submit.csv"):
    """
    For predicting de test set, saving a file for submition.
    Args:
    -parameters: result from training the NN
    -pricesstd: standard deviation of prices from training
    -pricesmean: mean of prices from training set
    """
    dataset = pd.read_csv("../data_filled_ready_to_predict.csv")

    states = dataset.groupby(["state_name"])

    results = {}
    #for g in states.groups.keys():
    state = states.get_group(g)
    ids = state[["id"]]
    places = one_hot_encode(state[["place_name"]].values, encodesize=542)
    expenses = standarize(state[["expenses"]].values)
    surfacetotal = standarize(state[["surface_total_in_m2"]].values)
    surfacecovered = standarize(state[["surface_covered_in_m2"]].values)
    proptypes = one_hot_encode(state[["property_type"]].values - 1, encodesize=6)
    xyz = coordenates_encode(state[["lat"]].values,state[["lon"]].values)
    #descriptions = state[["description"]]

    datastd = unite_subcols(expenses, surfacetotal)
    datastd = unite_subcols(datastd, surfacecovered)
    datastd = unite_subcols(datastd, proptypes)
    datastd = unite_subcols(datastd, places)
    datastd = unite_subcols(datastd, xyz)
    #datastd = unite_subcols(datastd, descriptions)

    predictions = predict(parameters, datastd.T, pricesstd, pricesmean)
    results = unite_subcols(ids, predictions.T)
    np.savetxt(filename, results, fmt='%i', delimiter=",")

if __name__ == '__main__':
    results = test_states(learning_rate=.5, epochs=3)
    #for g in results.keys():
    #    predict_test(results[g]['parameters'],
    #                 results[g]['pricestd'],
    #                 results[g]['pricemean'],filename=str(g)+'submit.csv')
