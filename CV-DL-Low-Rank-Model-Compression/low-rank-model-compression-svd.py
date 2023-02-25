import tensorflow_datasets as tfds
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.layers import Layer
from keras.optimizers import SGD
import scipy.ndimage as ndimage
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data
import seaborn as sn
import pandas as pd

# Ryan Filgas
# Computer Vision

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
#    if epoch < 4:
    return lr
#    else:
#      return lr * tf.math.exp(-.7)
callback = keras.callbacks.LearningRateScheduler(scheduler)
# model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
# model.compile(tf.keras.optimizers.SGD(), loss='mse')
# round(model.optimizer.lr.numpy(), 5)

def pre_process_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    xy = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
    image = tfa.image.translate(image, xy, interpolation='NEAREST')
    image = tfa.image.rotate(image, np.random.uniform(-15, 15), interpolation = 'NEAREST')
    return image, label

# Prefetched datasets
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

compression_level = .125
EPOCHS = 10
LEARNING_RATE = .013


ds_train = ds_train.map(pre_process_image, num_parallel_calls=tf.data.AUTOTUNE) #take out autotune
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE) #remove

ds_test = ds_test.map(pre_process_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_train = ds_test.prefetch(tf.data.AUTOTUNE) #remove

# INITIAL MODEL TRAINING
""" Done with model training. Saved the model for later loading. Comment back in to train a new model.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(50, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(10, activation='softmax', use_bias=True, bias_initializer="ones"),
])

sgd = SGD(0.02)
model.compile(
    optimizer= sgd,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model1_results = model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
)
model.save('pre_svd')
"""

model = tf.keras.models.load_model('pre_svd')
model.summary()

layers = list()

# PERFORM SVD AND OUTPUT NEW MATRICES
for layer in model.layers:
    if(layer.name != "flatten"):
        # this returned the kernel matrix and bias vector. We just want the matrix
        weights, bias_vec = layer.get_weights()

        # apply svd
        e, u, v = tf.linalg.svd(weights, compute_uv=True)
        num_features = np.int32(len(e)*compression_level)
        u = np.float32(u)
        e = np.float32(e)
        v = np.float32(v)

        # Slice U and E along columns
        u = u[:,:num_features]
        e = e[:num_features]

        # bias_vec = bias_vec[:num_features]
        #slice V along rows from the top
        v = v[:num_features,:]

        #convert to proper dims
        bias_vec = np.atleast_2d(bias_vec)
        # e = np.float64(np.atleast_2d(e))
        # get U' and add bias vec
        ue = np.dot(u, np.diag(e))
        # create random bias vec for other part and add it
        bias = np.random.uniform(-1,1,num_features).reshape(num_features, 1).T
        layers.append([ue, v, bias_vec, bias])

###############------MODEL2 Parameters
dense1, dense2, w1, w2 = layers[0]
dense3, dense4, w3, w4= layers[1]
dense5, dense6, w5, w6 = layers[2]

# Get shapes for each layer on the fly
layer_1_val = np.shape(dense1)[1]
layer_2_val = np.shape(dense2)[1]
layer_3_val = np.shape(dense3)[1]
layer_4_val = np.shape(dense4)[1]
layer_5_val = np.shape(dense5)[1]
layer_6_val = np.shape(dense6)[1]

# New compressed model
model2= tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28,1)),
    tf.keras.layers.Dense(layer_1_val, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(layer_2_val, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(layer_3_val, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(layer_4_val, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(layer_5_val, activation='relu', use_bias=True, bias_initializer="ones"),
    tf.keras.layers.Dense(layer_6_val, activation='softmax', use_bias=True, bias_initializer="ones"),
])


# Get shapes for setting weights.
w1_shape = np.shape(model2.layers[1].get_weights()[1])
w2_shape = np.shape(model2.layers[2].get_weights()[1])
w3_shape = np.shape(model2.layers[3].get_weights()[1])
w4_shape = np.shape(model2.layers[4].get_weights()[1])
w5_shape = np.shape(model2.layers[5].get_weights()[1])
w6_shape = np.shape(model2.layers[6].get_weights()[1])

#separate the bias layers we have and reshape them for transfer.
p1 = w1[0][0:w1_shape[0]].copy()
p3 = w3[0][0:w3_shape[0]].copy()
p5 = w5[0][0:w5_shape[0]].copy()

# Randomly initialize the weights we don't have
layer_2_bias = np.random.uniform(-1,1,w2_shape[0])
layer_4_bias = np.random.uniform(-1,1,w4_shape[0])
layer_6_bias = np.random.uniform(-1,1,w6_shape[0])

model2.layers[1].set_weights((dense1, p1))
model2.layers[2].set_weights((dense2, layer_2_bias))
model2.layers[3].set_weights((dense3, p3))
model2.layers[4].set_weights((dense4, layer_4_bias))
model2.layers[5].set_weights((dense5, p5))
model2.layers[6].set_weights((dense6, layer_6_bias))
x=2

# run model
sgd = SGD(LEARNING_RATE)
model2.compile(
    optimizer= 'adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

print(model2.summary())

model2_results = model2.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test,
    callbacks=[callback],
)
# model.save('pre_svd')

###############----- OUTPUT FOR MODEL 1
#PLOT RESULTS FOR PART 1
# # Plot Confusion Matrix
# y_predic = model.predict(ds_test)
# y_predic = np.argmax(y_predic,axis = 1)
# y_test = np.concatenate([y for x, y in ds_test], axis=0)
# columns = [0,1,2,3,4,5,6,7,8,9]
# confm = confusion_matrix(y_test, y_predic)
# df_cm = pd.DataFrame(confm, index=columns, columns=columns)
# ax = sn.heatmap(df_cm, annot=True, fmt="d")
# plt.xlabel("Truth")
# plt.ylabel("Predicted")
# plt.savefig("output.png")

# # Plot loss chart
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 100), model1_results.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 100), model1_results.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 100), model1_results.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 100), model1_results.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.savefig("output.png")



#### MODEL 2 RESULTS
# Plot Confusion Matrix
y_predic = model2.predict(ds_test)
y_predic = np.argmax(y_predic,axis = 1)
y_test = np.concatenate([y for x, y in ds_test], axis=0)
columns = [0,1,2,3,4,5,6,7,8,9]
confm = confusion_matrix(y_test, y_predic)
df_cm = pd.DataFrame(confm, index=columns, columns=columns)
ax = sn.heatmap(df_cm, annot=True, fmt="d")
plt.xlabel("Truth")
plt.ylabel("Predicted")
plt.savefig("4th_svd_uncompressed_cm.png")

# Plot loss chart
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 10), model2_results.history["loss"], label="train_loss")
plt.plot(np.arange(0, 10), model2_results.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 10), model2_results.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 10), model2_results.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("4th_svd_uncompressed_graph.png")