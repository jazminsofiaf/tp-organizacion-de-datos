import numpy as np
import pandas as pd


#np.random.seed(1)

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    return n_x, n_y

def initialize_parameters(n_x, n_h, n_y):
  """
  Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

  Returns:
    W1 -- weight matrix of shape (n_h, n_x)
    b1 -- bias vector of shape (n_h, 1)
    W2 -- weight matrix of shape (n_y, n_h)
    b2 -- bias vector of shape (n_y, 1)
  """
  
  W1 = np.random.randn(n_h, n_x) * 0.01
  b1 = np.zeros((n_h, 1))
  W2 = np.random.randn(n_y, n_h) * 0.01
  b2 = np.zeros((n_y, 1))

  #print "\ninicializando parametros"
  #print("W1.shape:", W1.shape)
  #print("b1.shape:", b1.shape)
  #print("W2.shape:", W2.shape)
  #print("b2.shape:", b2.shape)


  parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

  return parameters

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def relu(z):
    return np.maximum(z,0)

def customized_relu(z,cot_max,cot_min):
    print z
    return np.maximum(np.minimum(z,cot_max),cot_min)

def forward_propagation(X, parameters,cot_max,cot_min):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

   

    Z1 = np.dot(W1, X) + b1
    A1 = customized_relu(Z1,cot_max,cot_min)
    Z2 = np.dot(W2, A1) + b2
    #A2 = sigmoid(Z2)
    A2 = Z2

   #print "\nforward propagation"
    #print("Z1.shape:", Z1.shape)
    #print("A1.shape:", A1.shape)
    #print("Z2.shape:", Z2.shape)
    #print("A2.shape:", A2.shape)


    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost

    Arguments:
    A2 -- The output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    return mean_square_error(A2,Y)

def root_mean_square_error(A2,Y):
    cost = Y - A2
    cost = cost.mean()
    return cost  
    
def mean_square_error(A2,Y):
    cost = Y - A2
    cost = np.square(cost).mean()
    return cost

def loss(A2,Y):
    A2 = pd.Series(A2[0])
    Y = pd.Series(Y[0])

    return (-1*(A2.multiply(np.log(Y))).subtract(pd.Series([abs(1-a) for a in A2]).multiply(np.log([abs(1-y) for y in Y])))).mean()
   
def backward_propagation(parameters, cache, X, Y,cot_max,cot_min):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """

    
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    
    dZ2 =  (Y-A2)  
    dW2 = (1.0/m) * np.dot(dZ2, A1.T)
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)


    
    
    
    dZ2 = A2 - Y
    dW2 = (1.0/m) * np.dot(dZ2, A1.T)
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    Z1 = cache["Z1"]
    aux = (Z1*(Z1<=cot_max))
    dZ1 = np.dot(W2.T,dZ2)*(1*(aux>= cot_min))
    dW1 = (1.0/m) * np.dot(dZ1, X.T)
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)

    #print "\nbackward parametros"
    #print("dW1.shape:", dW1.shape)
    #print("db1.shape:", db1.shape)
    #print("dW2.shape:", dW2.shape)
    #print("db2.shape:", db2.shape)



    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
 
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}


    #print("\n\nupdating parameters\n")
    #print("W1:",W1)
    #print("b1:",b1.T[0])
    #print("W2:",W2[0])
    #print("b2:",b2.T[0])

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=True, parameters=None, learning_rate=None,cot_max=2.04, cot_min = -0.4):
    """
    Arguments:
    X -- dataset of shape (num_feautures, number of examples)
    Y -- labels of shape (1, number of examples)
   
    n_h -- size of the hidden layer
    
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x, n_y= layer_sizes(X, Y)

    if not parameters:
        parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        print("iteration number %i"%i)
        
        A2, cache = forward_propagation(X, parameters,cot_max,cot_min)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y,cot_max,cot_min)
        if learning_rate:
            parameters = update_parameters(parameters, grads, learning_rate)
        else:
            parameters = update_parameters(parameters, grads)
       
        # Print the cost every 1000 iterations
        if print_cost and i % 50 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X, pricestd, pricemean,cot_max,cot_min):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model
    """
    # Computes probabilities using forward propagation.
    A2, cache = forward_propagation(X, parameters,cot_max,cot_min)
    predictions = customized_relu(A2,cot_max,cot_min) * pricestd + pricemean
    return predictions


hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
def test_hidden_layers(X, Y, validation, labels):
  for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    #plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, validation, Y.std(), Y.mean())
    mse = np.square(labels - predictions).mean()
    #accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size))
    print ("Accuracy for {} hidden units: {}".format(n_h, mse))