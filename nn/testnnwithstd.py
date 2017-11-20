import numpy as np
import pandas as pd
from nn import *
from standarization import *

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

msk = np.random.rand(len(datastd)) < 0.8
train = datastd[msk]
test = datastd[~msk]
trainprices = pricesstd[msk]
testprices = prices[~msk]

sh = testprices.shape[0]
testprices = testprices.reshape((1,sh))
sh = trainprices.shape[0]
trainprices = trainprices.reshape((1,sh))

parameters = nn_model(train.T,trainprices,5,num_iterations=10000)

predictions = predict(parameters, test.T)

predictions = predictions.T * testprices.std() + testprices.mean()

print("predict: ", predictions)
print("real: ",testprices.T)
#accuracy = float((np.dot(testprices,predictions) + np.dot(1-testprices,1-predictions))/float(testprices.size)*100)
mse = np.square(testprices.T - predictions).mean()
print(mse)

#test_hidden_layers(trainabastoset.T,trainabastolabel)
