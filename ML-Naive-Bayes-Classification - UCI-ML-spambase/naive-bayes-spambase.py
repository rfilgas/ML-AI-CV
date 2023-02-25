import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import e
import seaborn as sn
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data

#Ryan Filgas

#Split data, permute it, and send back training and test sets
def split_data(x, train_percent):
    size = len(x) #get size of dataset
    train_set_num = int(size * train_percent) #get number of datapoints
    y = np.array(x).copy() #Don't edit original data
    np.random.shuffle(np.array(y))
    train = x[:train_set_num]
    test = x[train_set_num:]
    return pd.DataFrame(np.float64(train)), pd.DataFrame(np.float64(test))

#Use pandas and numpy to compute standard deviation
def m_dev(matrix_or_dataframe):
    x = pd.DataFrame(matrix_or_dataframe)
    mean = pd.DataFrame(x.sum(axis=0)/len(x[0]))
    #subtract from each data point its corresponding mean and square it.
    stdev = pd.DataFrame(np.array(x)-np.array(mean).T).pow(2)
    #sum up the data points, dicide by the number of data points, take the square root.
    stdev = pd.DataFrame(np.sqrt(stdev.sum(axis=0)/len(x[0])))
    return pd.DataFrame(np.float64(mean)), pd.DataFrame(np.float64(stdev))

#vectorization of e^x
def e_pow(x):
    return np.exp(x)
e_pow_vec = np.vectorize(e_pow)

#vectorization of x^-1 to compute 1/x for each element
def inverse_pow(x):
    if(x == 0):
        return x
    return x ** -1
inverse_pow_vec = np.vectorize(inverse_pow)

# p(xi | cj) = N(input, mean of class, standard deviation of class)
def N_pdf(x, mean, stdev):
    x, mean, stdev = np.array(x), np.array(mean), np.array(stdev)
    denominator = inverse_pow_vec((stdev * np.sqrt(2*np.pi))) # 1/(std * sqrt(2*pi))
    exponent_term = ((np.array(x) - np.array(mean)) ** 2) / ((stdev ** 2) * 2) #((x-mean)^2 / 2(std^2))
    exponent = (e_pow_vec((-1 * exponent_term))) #e^(-x)
    exponent[exponent == 0] = .000001 #replace 0s
    result = denominator * exponent #multiply the two terms
    result[result == 0] = .000001 #replace 0s again
    return result

#For pretty printing.
float_formatter = "{:.4f}".format
KEY_LOCATION = 57

################################################################################

#PART 1: Split training and test data

#Import and convert data
data= pd.DataFrame(pd.read_csv('spambase.data', sep=",", header=None))
spam = data[data[57] == 1] #isolate spam
not_spam = data[data[57] == 0]#isolate not spam
spam = np.delete(np.array(spam),(KEY_LOCATION), axis=1) #drop keys
not_spam = np.delete(np.array(not_spam),(KEY_LOCATION), axis=1) #drop keys
spam_train, spam_test = split_data(spam, .5) #permute and split data >>
not_spam_train, not_spam_test = split_data(not_spam, .5) #into train and test

#Set zeros to .0001 to prevent divide by 0 errors in log functions.
spam_train.replace(to_replace = 0, value = .000001, inplace=True)
not_spam_train.replace(to_replace = 0, value = .000001, inplace=True)
spam_test.replace(to_replace = 0, value = .000001, inplace=True)
not_spam_test.replace(to_replace = 0, value = .000001, inplace=True)

################################################################################

#PART 2 Create probabilistic model.

# 2A. Compute the prior probability for each class 1 (spam) and 0 (not-spam) in the training data. 
# As described in part 1, P(1) should be about 0.4. 

# Train set prior probability
train_spam_prob = len(spam_train) / (len(spam_train) + len(not_spam_train))
train_not_spam_prob = len(not_spam_train) / (len(spam_train) + len(not_spam_train))

# Test set prior probability
test_spam_prob = len(spam_test) / (len(spam_test) + len(not_spam_test))
test_not_spam_prob = len(not_spam_test) / (len(spam_test) + len(not_spam_test))

# 2B. 
#Get mean and standard dev for each group of data, replacing 0 with .0001
train_spam_mean, train_spam_stdev = m_dev(spam_train)
train_not_spam_mean, train_not_spam_stdev = m_dev(not_spam_train)

################################################################################

# 3. Run Naïve Bayes on the test data.

###Spam Bayes
n_spam_test = N_pdf(spam_test, train_spam_mean.T, train_spam_stdev.T) #pdf
n_spam_test = np.log2(np.array(pd.DataFrame(n_spam_test)))
n_spam_test = pd.DataFrame(n_spam_test).sum(axis=1) + np.log2(test_spam_prob) #sum and add prior

n2_spam_test = N_pdf(spam_test, train_not_spam_mean.T, train_not_spam_stdev.T) #pdf
n2_spam_test = np.log2(np.array(pd.DataFrame(n2_spam_test)))
n2_spam_test = pd.DataFrame(n2_spam_test).sum(axis=1) + np.log2(test_not_spam_prob) #sum and add prior

###spam prediction calculations
response = pd.DataFrame([n_spam_test, n2_spam_test]).T
response.columns = ['Spam', 'NotSpam']
response['Prediction'] = 1
response.loc[(response['NotSpam'] > response['Spam']), 'Prediction'] = 0

sp_correct = len(response.loc[response['Prediction'] == 1])
sp_total = len(response)

#####################################################################################

###Not spam Bayes
n_not_spam_test = N_pdf(not_spam_test, train_not_spam_mean.T, train_not_spam_stdev.T) #pdf
n_not_spam_test = np.log2(np.array(pd.DataFrame(n_not_spam_test)))
n_not_spam_test = pd.DataFrame(n_not_spam_test).sum(axis=1) + np.log2(test_not_spam_prob) #sum and add prior

n2_not_spam_test = N_pdf(not_spam_test, train_spam_mean.T, train_spam_stdev.T) #pdf
n2_not_spam_test = np.log2(np.array(pd.DataFrame(n2_not_spam_test)))
n2_not_spam_test = pd.DataFrame(n2_not_spam_test).sum(axis=1) + np.log2(test_spam_prob) #sum and add prior

###Not spam prediction calculations
response2 = pd.DataFrame([n_not_spam_test, n2_not_spam_test]).T
response2.columns = ['Spam', 'NotSpam']
response2['Prediction'] = 0
response2.loc[(response2['NotSpam'] <= response2['Spam']), 'Prediction'] = 1

nsp_correct = len(response2.loc[response2['Prediction'] == 0])
nsp_total = len(response2)

#####################################################################################
#REPORT RESULTS:

spam = np.array(response['Prediction'])
spam_truth = np.ones(907)
not_spam = np.array(response2['Prediction'])
not_spam_truth = np.zeros(1394)
test_x = np.append(spam, not_spam)
test_y = np.append(spam_truth, not_spam_truth)
precision = sp_correct / sp_total
recall = sp_correct / (sp_correct + (nsp_total - nsp_correct))

print("Training Spam Prior: ", float_formatter(train_spam_prob))
print("Training Not Spam Prior: ", float_formatter(train_not_spam_prob))
print("Test Spam Prior: ", float_formatter(test_spam_prob))
print("Test Not Spam Prior: ", float_formatter(test_not_spam_prob))
print("Spam Accuracy: ", float_formatter(sp_correct/sp_total))
print("Not Spam Accuracy: ", float_formatter(nsp_correct/nsp_total))
print("Total Accuracy", float_formatter(((sp_correct + nsp_correct) / (sp_total + nsp_total))))
print("Precision: ",float_formatter(precision) )
print("Recall: ", float_formatter(recall))

####################
#CONFUSION MATRIX
print("Number of test data points:", np.size(test_y))
cm = confusion_matrix(test_y, test_x)
plt.figure(figsize = (2,2))
sn.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
