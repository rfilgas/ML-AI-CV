from cgitb import grey
from black import out
from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ryan Filgas
# Computer Vision


def add_padding(img, padding_size):
    width = np.shape(img)[0]
    height = np.shape(img)[1]
    vertical = np.array(np.zeros((width, padding_size)))
    img1 = np.append(vertical, img, axis=1)
    img1 = np.append(img1, vertical, axis=1)
    length = height + (padding_size * 2)
    horizontal = np.array(np.zeros((padding_size, length)))
    img1 = np.append(img1, horizontal, axis=0)
    img1 = np.append(horizontal, img1, axis=0)
    return np.array(img1)


def convolve(img, filter, stride):
    result = np.zeros(np.shape(img))
    size = np.shape(filter)[0]
    reach = int(np.shape(filter)[0]/2)
    left_bound = int(size/2)
    top_bound = int(size/2)
    right_bound = np.shape(np.array(img))[0] - left_bound
    bottom_bound = np.shape(np.array(img))[1] - left_bound

    for i in range(left_bound, right_bound, stride):
        for j in range(top_bound, bottom_bound, stride):
            matrix = img[i-reach:i+reach+1, j-reach:j+reach+1]
            result[i][j] = np.sum(np.multiply(matrix, filter))
    # don't return the padding
    return result[1:np.shape(img)[0]-1, 1:np.shape(img)[1]-1]


def convolve_optical_flow(A_1, A_2, identity_t, stride, conv_width):
    result = np.zeros((np.shape(A_1)[0], np.shape(A_1)[1], 2))

    # set bounds
    size = conv_width
    reach = int(size/2)
    left_bound = int(size/2)
    top_bound = int(size/2)
    right_bound = np.shape(np.array(A_1))[0] - left_bound
    bottom_bound = np.shape(np.array(A_1))[1] - left_bound

    # convolve
    for i in tqdm(range(left_bound, right_bound, stride)):
        for j in range(top_bound, bottom_bound, stride):

            # save ~10 calculations for each pixel.
            leftx = i-reach
            rightx = i + reach + 1
            lefty = j-reach
            righty = j+reach+1
            length = size * size

            # extract the matrices
            c1 = np.reshape(
                A_1[leftx:rightx, lefty:righty], length)
            c2 = np.reshape(
                A_2[leftx:rightx, lefty:righty], length)
            It = np.reshape(
                identity_t[leftx:rightx, lefty:righty], length)

            A = np.reshape(np.stack((c1, c2), axis=0), (9, 2))
            AT = A.T
            ATA_inv = np.linalg.pinv(np.matmul(AT, A))
            ATIt = np.matmul(AT, It.T)
            goal = np.matmul(ATA_inv, ATIt)
            result[i][j] = goal
    # don't return the padding
    return result[1:np.shape(A_1)[0]-1, 1:np.shape(A_1)[1]-1]


# Input images
frame1_a = cv2.imread("frame2_a.png", cv2.IMREAD_GRAYSCALE)
frame1_b = cv2.imread("frame2_b.png", cv2.IMREAD_GRAYSCALE)

img_stride = 1
img_filter_size = 3
img_padding = 1
img_padded_1 = add_padding(frame1_a, img_padding)
img_padded_2 = add_padding(frame1_b, img_padding)

dogx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
dogy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# ****************** GET DoG X and Y Filters for both images

img1_dogx = convolve(img_padded_1, dogx, img_stride)
img1_dogy = convolve(img_padded_1, dogy, img_stride)

img2_dogx = convolve(img_padded_2, dogx, img_stride)
img2_dogy = convolve(img_padded_2, dogy, img_stride)

# ****************** GET DoG X and Y Filters for Image 1


# ********* GET SOBEL FILTERS FOR BOTH IMAGES to Calc Identity

img1_sobel = np.sqrt(img1_dogx**2 + img1_dogy**2)
img2_sobel = np.sqrt(img2_dogx**2 + img2_dogy**2)


identity_t = - (img2_sobel - img1_sobel)
# ******** GET SOBEL FILTERS FOR BOTH IMAGES to Calc Identity


# *************************************** Add Padding

img1_dogx_padded = add_padding(img1_dogx, img_padding)
img1_dogy_padded = add_padding(img1_dogy, img_padding)
identity_t_padded = add_padding(identity_t, img_padding)

# *************************************** Add Padding

catch = convolve_optical_flow(img1_dogx_padded, img1_dogy_padded,
                              identity_t_padded, img_stride, img_filter_size)


# *************************************** GRAPH 1
output = cv2.cvtColor(frame1_b, cv2.COLOR_BGR2RGB)
thickness = 1
color = (0, 255, 0)
count = 0
x_len = np.shape(frame1_a)[0]
y_len = np.shape(frame1_a)[1]
catch = np.reshape(catch, (np.shape(frame1_a)[0], np.shape(frame1_a)[1], 2))

for i in range(2, x_len, 2):
    for j in range(2, y_len, 2):
        x = catch[i][j][0]
        y = catch[i][j][1]
        magnitude = np.sqrt(x**2 + y**2)

        if magnitude > .8:

            start_point = (j, i)
            end_point = (int(j + x), int(i + y))
            cv2.line(output, start_point, end_point,
                     color, thickness)
            thickness = 1
            color = (0, 255, 0)

# *************************************** GRAPH 2
output2 = cv2.cvtColor(frame1_a, cv2.COLOR_BGR2RGB)
thickness = 2
color = (0, 255, 0)
count = 0
x_len = np.shape(frame1_a)[0]
y_len = np.shape(frame1_a)[1]
catch = np.reshape(catch, (np.shape(frame1_a)[0], np.shape(frame1_a)[1], 2))

for i in range(1, x_len, 1):
    for j in range(1, y_len, 1):
        x = catch[i][j][0]
        y = catch[i][j][1]
        magnitude = np.sqrt(x**2 + y**2)
        green = int((magnitude * (255/10)))  # amplify signal
        if(green > 255):  # clip pixels to 255
            green = 255
        color = (0, green, 0)

        # graph it
        start_point = (j, i)
        end_point = (j, i)
        cv2.line(output2, start_point, end_point,
                 color, thickness)

        # reset vars for next loop
        thickness = 1
        color = (0, 255, 0)
        count += 1

plt.imshow(output)
plt.show()

plt.imshow(output2)
plt.show()

cv2.imwrite("image.jpg", output)
cv2.imwrite("image2.jpg", output2)
