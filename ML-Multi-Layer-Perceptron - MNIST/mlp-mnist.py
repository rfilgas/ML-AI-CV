import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from math import e
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data


df_train_set = np.float64(pd.read_csv("mnist_train.csv").to_numpy())
df_test_set = np.float64(pd.read_csv("mnist_test.csv").to_numpy())
df_train = df_train_set.copy()
df_test = df_test_set.copy()


#Data Inputs
INPUTS = np.size(df_train, 1) #785 x 1
FEATURES = df_train.shape[0] #60000 x 1
TEST_FEATURES = df_test.shape[0] #10000 x 785
TEST_TRUTH = np.array(df_test[0]).copy()

#Training Parameters
EPOCHS = 50
PERCEPTRONS = 10
LEARNING_RATE = np.float64(.1)
ALPHA = .9
NUM_HIDDEN_WEIGHTS = 100
BIAS_VAL = 1
SIGMA_BIAS = .9
ZERO_VAL = .1
WEIGHT_LOW = -.05
WEIGHT_HIGH = .05


#Helper functions:##########################################
#This function group is for vector creation. Each function 
# takes an input for size and outputs a vector of specified type.
def one_vec(size):
    return np.reshape(np.full((size,1), 1), (size, 1))
def learning_vec(size):
    return np.reshape(np.full((size,1), LEARNING_RATE), (size, 1))
def alpha_vec(size):
    return np.reshape(np.full((size,1), ALPHA), (size, 1))
def zero_vec(size):
    return np.reshape(np.full((size,1), 0), (size, 1))


#This function group is for sigmoid squashing.
def sigmoid(z):
    if z < 0:
        return np.float64(pow(e,z))/np.float64((1+pow(e,z)))
    else:    
        var1 = np.float64(pow(e, -z))
        return (np.float64(1))/(np.float64(1) + var1)
sigmoid_vec = np.vectorize(sigmoid)



#This function group sets onehot encoding. 
def new_one_hot_vec(input, size):
    #x = np.float64(np.zeros(size).reshape((size,1)))
    x = np.reshape(np.float64(np.full(size, ZERO_VAL)), (size,1))
    x[np.int64(input)] = SIGMA_BIAS
    return x



#This function group returns a new permuted matrix and key set.
def permute_train_data(matrix):
    new_matrix = (np.array(matrix)).copy()
    np.random.shuffle(new_matrix)
    key = np.array((np.transpose(new_matrix))[0]).copy()
    key = np.reshape(key, (FEATURES, 1))
    new_matrix = np.float64(new_matrix)/np.float64(255)
    new_matrix[:, 0] = np.float64(BIAS_VAL)
    return(new_matrix, key)


#############################################################


#setup
hidden_weights = np.float64(np.random.uniform(low=WEIGHT_LOW, high=WEIGHT_HIGH, size=((NUM_HIDDEN_WEIGHTS+1),INPUTS))) #Nx785
output_weights = np.float64(np.random.uniform(low=WEIGHT_LOW, high=WEIGHT_HIGH, size=(PERCEPTRONS,(NUM_HIDDEN_WEIGHTS+1)))) #10x785
output_weights[:,0] = BIAS_VAL
hidden_weights[:,0] = BIAS_VAL


#normalize and set test data. train data will be done via function.
test_target = np.float64(np.array(np.transpose(df_test_set)[0]).copy()) #1x785
df_test = np.float64(df_test)/np.float64(255)
df_test[:, 0] = np.float64(BIAS_VAL)


#testing data
train_accuracy = np.float64(np.empty([EPOCHS, 1]))
test_accuracy = np.float64(np.empty([EPOCHS, 1]))

# hidden deltas are initated and stored from prior iterations
prev_hidden_delta = np.zeros([NUM_HIDDEN_WEIGHTS+1, 1], dtype = np.float64())
prev_output_delta = np.zeros([PERCEPTRONS, 1], dtype = np.float64())



##############################################
#TRAINING
##############################################

for i in range(0,EPOCHS):
    print("Epoch: ", i, " started.")
    #get the new training data and target vector
    df_train, train_target = permute_train_data(df_train_set)

    accuracy_temp = 0
    absolute_prediction = 0

    for k, feature in enumerate(df_train):
        
        #FORWARDPROP
        #(1) Forward pass, compute h_j, activations of hidden neurons and o_k, activations of output neurons. 
        target_vec = new_one_hot_vec(np.int64(train_target[k]),PERCEPTRONS) #this is the onehot target vector

        feature_temp = np.reshape(np.array(np.transpose(feature)).copy(), (INPUTS,1)) #reformat input
        hidden_activation = sigmoid_vec(np.matmul(hidden_weights, feature_temp)) #multiply and squash
        hidden_activation[0] = np.float64(1) #reset bias
        hidden_activation = np.reshape(hidden_activation, (NUM_HIDDEN_WEIGHTS+1,1)) #reformat

        output_activation = sigmoid_vec(np.matmul(output_weights, hidden_activation)) #multiply and squash
        output_activation = np.reshape(output_activation, (PERCEPTRONS,1)) #reformat
       
        predicted_val = np.argmax(output_activation)#get predictions
        target_val = np.int64(train_target[k][0])

        #BACKPROP
        #(2) Compute error (deltas) for output neurons: delta_k's
        output_delta = output_activation * ((one_vec(PERCEPTRONS)-output_activation) * (target_vec - output_activation))

        #(3) Compute error (deltas) for hidden neurons: delta_j's
        mysum = np.matmul(np.transpose(output_weights), output_delta)
        hidden_delta = hidden_activation * (one_vec(NUM_HIDDEN_WEIGHTS+1) - hidden_activation) * mysum

        #(4) Compute weight updates for hidden-to-output edges
        output_weight_update = (LEARNING_RATE * output_delta * np.transpose(hidden_activation)) + (ALPHA * prev_output_delta)
        output_weights = output_weights + output_weight_update

        #(5) Compute weight updates for input-to-hidden edges 
        hidden_weight_update = np.transpose((LEARNING_RATE * np.transpose(hidden_delta) * np.reshape(feature, (INPUTS, 1)))) + (ALPHA * prev_hidden_delta)
        hidden_weights = hidden_weights + np.reshape(hidden_weight_update, (NUM_HIDDEN_WEIGHTS+1, INPUTS))
        
        #save previous weight updates
        prev_hidden_delta = np.array(hidden_weight_update).copy()
        prev_output_delta = np.array(output_weight_update).copy()

        #count a correct prediction
        accuracy_temp += (1 if predicted_val == target_val else 0)

    # take a percentage of correct predictions out of total predictions and assign to array for each epoch
    train_accuracy[i] = np.float64(np.float64(accuracy_temp) / np.float64(FEATURES)*100)





    ##############################################
    #TESTING
    ##############################################

    #collect varuables for plotting
    test_x = np.empty([TEST_FEATURES, 1])
    test_y = np.empty([TEST_FEATURES, 1])
    test_accuracy_temp = 0
    test_prediction = 0

    for j, feature in enumerate(df_test):

    #FORWARDPROP
    #(1) Forward pass, compute h_j, activations of hidden neurons and o_k, activations of output neurons.
        #t_target_vector = new_one_hot_vec(np.float64(train_target[k].copy()),PERCEPTRONS) #this is the onehot target vector
        target_vec = new_one_hot_vec(np.int64(test_target[j]),PERCEPTRONS) #this is the onehot target vector
        feature_temp = np.reshape(np.array(np.transpose(feature)).copy(), (INPUTS,1)) #reformat input
        hidden_activation = sigmoid_vec(np.matmul(hidden_weights, feature_temp)) #multiply and squash
        hidden_activation[0] = np.float64(1) #reset bias
        hidden_activation = np.reshape(hidden_activation, (NUM_HIDDEN_WEIGHTS+1,1)) #reformat

        output_activation = sigmoid_vec(np.matmul(output_weights, hidden_activation)) #multiply and squash
        output_activation = np.reshape(output_activation, (PERCEPTRONS,1)) #reformat
       
        #tabulate results
        predicted_val = np.argmax(output_activation)#get predictions
        target_val = np.int64(test_target[j])
        test_accuracy_temp += (1 if predicted_val == target_val else 0)
        test_x[j] = target_val
        test_y[j] = predicted_val

    #add results by epoch and give a status message.
    test_accuracy[i] = (test_accuracy_temp / TEST_FEATURES)*100
    print("TRAIN: ", train_accuracy[i])
    print("TEST: ", test_accuracy[i])


####################
#LINE GRAPH
epoch_list = list(range(1, EPOCHS+1))
plt.plot(epoch_list, train_accuracy)
plt.plot(epoch_list, test_accuracy)
plt.show()


####################
#CONFUSION MATRIX
cm = confusion_matrix(test_y, test_x)
plt.figure(figsize = (10,10))
sn.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()