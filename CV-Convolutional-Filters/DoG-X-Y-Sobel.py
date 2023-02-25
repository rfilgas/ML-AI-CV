from cgitb import grey
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


gaussian_1 = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float64)
gaussian_2 = 1/273 * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]], np.float64)

g1_filter_size = np.shape(gaussian_1)[0]
g1_padding = 1
filter1_og = cv2.imread("filter1_img.jpg", cv2.IMREAD_GRAYSCALE)
filter1 = add_padding(filter1_og, g1_padding)

g2_filter_size = np.shape(gaussian_2)[0]
g2_padding = 2
filter2_og = cv2.imread("filter2_img.jpg", cv2.IMREAD_GRAYSCALE)
filter2 = add_padding(filter2_og, g2_padding)


STRIDE = 1


# test = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
#                 [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], np.float64)

#### RUN FOR 3x3 and 5x5 Filters!#############################

result3x3 = convolve(filter1, gaussian_1, STRIDE)
result5x5 = convolve(filter1, gaussian_2, STRIDE)

cv2.imwrite('test.jpg', result3x3)


plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(filter1_og, cmap='gray')
plt.title("#nofilter")
plt.subplot(142)
plt.imshow(result3x3, cmap='gray')
plt.title("3x3 filter")
plt.subplot(143)
plt.imshow(result5x5, cmap='gray')
plt.title("5x5 filter")
plt.show()
# plt.imshow(filter1, cmap='gray')
# plt.imshow(result, cmap='gray')

result3x3 = convolve(filter2, gaussian_1, STRIDE)
result5x5 = convolve(filter2, gaussian_2, STRIDE)
plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(filter2_og, cmap='gray')
plt.title("#nofilter")
plt.subplot(142)
plt.imshow(result3x3, cmap='gray')
plt.title("3x3 filter")
plt.subplot(143)
plt.imshow(result5x5, cmap='gray')
plt.title("5x5 filter")
plt.show()

########## RUN for DoG Filters ####################################################################

# dogx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# dogy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
# dog_padding = 1
# dog_stride = 1
# dog_test_1 = add_padding(filter1_og, dog_padding)
# dog_test_2 = add_padding(filter2_og, dog_padding)

# result_dogx_1 = convolve(dog_test_1, dogx, dog_stride)
# result_dogy_1 = convolve(dog_test_1, dogy, dog_stride)

# result_dogx_2 = convolve(dog_test_2, dogx, dog_stride)
# result_dogy_2 = convolve(dog_test_2, dogy, dog_stride)

# cv2.imwrite('omg1.jpg', result_dogx_1)
# cv2.imwrite('omg2.jpg', result_dogx_2)

# plt.figure(figsize=[20, 5])
# plt.subplot(141)
# plt.imshow(filter1_og, cmap='gray')
# plt.title("#nofilter")
# plt.subplot(142)
# plt.imshow(result_dogx_1, cmap='gray')
# plt.title("g(x) filter")
# plt.subplot(143)
# plt.imshow(result_dogy_1, cmap='gray')
# plt.title("g(y) filter")
# plt.show()

# plt.figure(figsize=[20, 5])
# plt.subplot(141)
# plt.imshow(filter2_og, cmap='gray')
# plt.title("#nofilter")
# plt.subplot(142)
# plt.imshow(result_dogx_2, cmap='gray')
# plt.title("g(x) filter")
# plt.subplot(143)
# plt.imshow(result_dogy_2, cmap='gray')
# plt.title("g(y) filter")
# plt.show()

########## RUN for Sobel Filters ####################################################################
"""
dogx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
dogy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
dog_padding = 1
dog_stride = 1
dog_test_1 = add_padding(filter1_og, dog_padding)
dog_test_2 = add_padding(filter2_og, dog_padding)


dog_x_sobel = convolve(dog_test_1, dogx, dog_stride)
dog_y_sobel = convolve(dog_test_1, dogy, dog_stride)

dog_x_sobel_2 = convolve(dog_test_2, dogx, dog_stride)
dog_y_sobel_2 = convolve(dog_test_2, dogy, dog_stride)


sobel_1 = np.sqrt(dog_x_sobel**2 + dog_y_sobel**2)
sobel_2 = np.sqrt(dog_x_sobel_2**2 + dog_y_sobel_2**2)

plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(filter1_og, cmap='gray')
plt.title("#nofilter")
plt.subplot(142)
plt.imshow(sobel_1, cmap='gray')
plt.title("sobel filter")
plt.show()


plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(filter2_og, cmap='gray')
plt.title("#nofilter")
plt.subplot(142)
plt.imshow(sobel_2, cmap='gray')
plt.title("sobel filter")
plt.show()
"""
