import os
import time
import tensorflow as tf
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import math
import scipy
import keras
from matplotlib import pyplot as plt
from keras import layers, models, utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from keras.applications import InceptionResNetV2
from sklearn.metrics import classification_report, confusion_matrix

# Ryan Filgas
# COmputer Vision

pre_model=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(150,150,3))

############################# PRINT LAYERS ##################
"""
# get layer composition
for layer in pre_model.layers:
	filters = layer.get_weights()
	print(layer.name, np.shape(filters))
"""
#############################################################

############################# PRINT LAYER 1 FILTERS #########
"""
# get filter layer weights and normalize
filters = pre_model.layers[1].get_weights()

# normalize to 0-255
weight_min, weight_max = np.min(filters), np.max(filters)
filters = (filters - weight_min) / (weight_max - weight_min)
filters = np.int64(filters * 255)
filters = np.clip(filters, 0, 255).astype("uint8")

# Print each filter
index = 1
for j in range(32):
	ax = plt.subplot(4, 8, index)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.imshow(filters[0][:,:,:,j])
	index += 1
plt.savefig('filters.png')
"""
# Plot model for curiosity
# Plot_model(pre_model, to_file='model_test.png', show_shapes=True,show_layer_names=True)
#############################################################

############################# Load and Augment data #########
# Creating DataGen instances

BATCH_SIZE=64
EPOCHS=7

training_set = ImageDataGenerator(rescale=1/255.,
    samplewise_center=True, #mean centering
    samplewise_std_normalization=True, #feature normalization
    rotation_range= .2, #rotation
    width_shift_range=0.2, #width shift
    height_shift_range=0.2, #height shift
    horizontal_flip=True, #flip
)
test_set = ImageDataGenerator(rescale=1. / 255)

data_train = training_set.flow_from_directory(directory='cats_dogs_dataset/dataset/training_set',
                                                                     target_size=(150, 150),
                                                                     class_mode='binary',
                                                                     batch_size=BATCH_SIZE)
data_test = test_set.flow_from_directory(directory='cats_dogs_dataset/dataset/test_set',
                                                                     target_size=(150, 150),
                                                                     class_mode='binary',
                                                                     batch_size=2000,
                                                                     shuffle=False,
                                                                     )

subnet = tf.keras.Model(inputs = pre_model.input, outputs=pre_model.get_layer('conv2d_15').output)
#subnet.summary()

# # Part 1: pre_model - comment out part 2 and comment in part 1 to use.
# pre_model.trainable=False
# model = models.Sequential()
# model.add(pre_model)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()

# Part 2 Subnet
subnet.trainable=False
model = models.Sequential()
model.add(subnet)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

#compile and fit model
model.compile(optimizer='adam', loss='binary_crossentropy',
 	           metrics=['accuracy'])
model.fit(
    data_train, epochs=EPOCHS, validation_data=data_test,
)

results = model.evaluate(data_test)
print("test loss, test acc:", results)

Y_pred = model.predict(data_test, 2000 // BATCH_SIZE+1)
Y_pred = Y_pred.flatten()
Y_pred = np.where(Y_pred > 0.5, 1, 0)

print('Confusion Matrix')
print(confusion_matrix(data_test.classes, Y_pred))
print('Classification Report')
target_names = ['cats', 'dogs']
print(classification_report(data_test.classes, Y_pred, target_names=target_names))