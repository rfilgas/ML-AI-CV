
from lib2to3.pgen2.pgen import generate_grammar
import tensorflow as tf
from tensorflow import keras
from silence_tensorflow import silence_tensorflow
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, save_img, img_to_array
from IPython.display import Image, display
from keras import models
from keras import backend as K
import matplotlib.pyplot as plt
import pprint
import matplotlib.cm as cm
import numpy as np
import cv2
from matplotlib.pyplot import figure
tf.get_logger().setLevel('ERROR')
silence_tensorflow()

#Ryan FIlgas
#Computer Vision

def generate_map(input_arr, gradient_mod, pred_index):
    with tf.GradientTape() as tape:
        conv_layer_output, my_predictions = gradient_mod(input_arr)
        if pred_index == None: # to get other top classes
            pred_index = tf.argmax(my_predictions[0])
        class_channel = my_predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_layer_output)
    # Formula 1: calculate mean intensity by summing the
    # gradients along the featuremap and dividing by 14x14
    alpha_ck = tf.reduce_sum(grads, axis=(0, 1, 2)) / (14*14)
    #Formula 2: Multiply output conv_layer * gradient weights then sum and apply a relu.
    Ak = (conv_layer_output[0] * alpha_ck)
    sum_alpha_k = tf.reduce_sum((Ak),axis=(2))
    activation_map = tf.squeeze(tf.nn.relu(sum_alpha_k))
    return activation_map


def process_image(image):
    input_arr = tf.keras.preprocessing.image.img_to_array(image) #convert to array
    input_arr = np.array([input_arr])  # Convert to a batch
    input_arr = preprocess_input(input_arr)
    predictions = model.predict(input_arr)
    decoded = decode_predictions(predictions, top=3)
    x1 = K.tf.where((decoded[0][0])[2] == predictions[0]).numpy()[0][0]
    x2 = K.tf.where((decoded[0][1])[2] == predictions[0]).numpy()[0][0]
    x3 = K.tf.where((decoded[0][2])[2] == predictions[0]).numpy()[0][0]
    predictions = [x1,x2,x3]
    return predictions, input_arr, decoded

# # SAVED MODEL
# pre_model = tf.keras.applications.VGG16(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
# )
# pre_model.save('pre_model')

model = tf.keras.models.load_model('pre_model')

#######################################################################################################
# INITIAL CODE TO GET DECODED PREDICTIONS
data_set = tf.keras.preprocessing.image_dataset_from_directory(
    'images/input',
    label_mode=None,
    image_size=(224, 224),
    batch_size=1
).map(lambda x: (preprocess_input(x)))

x = model.predict(data_set)
pprint.pprint(decode_predictions(x, top=3))
top1 = K.argmax(x, axis=1)
top1 = np.argmax(x, axis=1)
print(top1)

####################################################################################
IMAGE_NUMBER = 5
last_layer=model.get_layer('block5_conv3')
image_path = "images/input/gc" + str(IMAGE_NUMBER) + ".jpg"
directory = "Output/0" + str(IMAGE_NUMBER) +"/"
image_og = tf.keras.preprocessing.image.load_img(image_path)
image_size = np.shape(image_og) #get this interpolation for later

image = tf.keras.preprocessing.image.load_img(image_path,
                                    target_size=(224,224)) #load at expected image size

predictions, input_arr, decoded = process_image(image)

# create a new model to use with gradient tape
gradient_mod = keras.models.Model([model.inputs],
[last_layer.output, model.output])

figure(figsize=(80, 60), dpi=72)

maps = list()
for i in range(len(predictions)):
    activation_map = generate_map(input_arr, gradient_mod, predictions[i])
    # get original image for overlay
    img = cv2.imread(image_path)
    image_shape = img.shape
    # normalize
    heatmap = tf.maximum(activation_map, 0) / tf.math.reduce_max(activation_map)
    # Resize the heatmap to fit the image 0 Interliniear means bilinear.
    heatmap = cv2.resize(np.float32(heatmap), (image_shape[1], image_shape[0]), interpolation= cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_VIRIDIS)
    output = cv2.addWeighted(img, .4, heatmap, 1 - .4, 0)

    # Plot images
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(img[:,:,::-1])
    ax1.set_title("Image: " + str(IMAGE_NUMBER))
    ax2.set_title("Pred " + str(i+1) + ":" + str((decoded[0][i])[1]))
    ax2.imshow(output[:,:,::-1])
    ax3.set_title("Saliency Map")
    ax3.imshow(heatmap[:,:,::-1])
    plt.savefig((directory + str(i+1) + ".jpg"))
