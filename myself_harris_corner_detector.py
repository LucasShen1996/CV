import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import corner_harris, corner_peaks
from skimage.color import *


# Harris corner detection
def my_harris_corner_detector(image,alpha , threshold):
    # Sobel
    def Sobel_filtering(gray):
        # get shape
        height, width = gray.shape
        # sobel windows
        M_y = np.array(((1, 2, 1),(0, 0, 0),(-1, -2, -1)), dtype=np.float64)
        M_x = np.array(((1, 0, -1),(2, 0, -2),(1, 0, -1)), dtype=np.float64)
        # padding
        tmp = np.pad(gray, (1, 1), 'edge')
        # prepare the empty I
        I_x = np.zeros((height,width), dtype=np.float64)
        I_y = np.zeros((height,width), dtype=np.float64)
        # get differential
        for y in range(height):
            for x in range(width):
                I_x[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * M_x)
                I_y[y, x] = np.mean(tmp[y: y + 3, x: x + 3] * M_y)
        I_x2 = I_x * I_x
        I_y2 = I_y * I_y
        I_xy = I_x * I_y

        return I_x2, I_y2, I_xy

    # gaussian filtering
    def gaussian_filtering(I, window_size, sigma):
        # get shape
        height, width = I.shape
        #padding
        I_t = np.pad(I, (1, 1), 'edge')
        # gaussian  filtering
        window = np.zeros((window_size, window_size), dtype=np.float32)
        for x in range(window_size):
            for y in range(window_size):
                window[y, x] = np.exp(-(x * x + y * y) / (2 * (sigma * sigma)))
        window = window / (sigma * sigma * 2 * np.pi)
        window = window / window.sum()

        for y in range(height):
            for x in range(width):
                I[y, x] = np.sum(I_t[y: y + window_size, x: x + window_size] * window)

        return I

    # corner detect
    def corner_detect(I_x2, I_y2, I_xy, alpha, threshold):

        height, width = I_xy.shape
        # get R
        R = (I_x2 * I_y2 - I_xy * I_xy ) - alpha * ((I_x2 + I_y2) * (I_x2 + I_y2) )

        # detect corner
        coner_loc = []
        for y in range(height):
            for x in range(width):
                if R[y][x] >= np.max(R) * threshold:
                    coner_loc.append([y, x])
        coner_loc = np.array(coner_loc)
        return coner_loc

    # change to gray image
    if len(image.shape)!= 2:
        image = rgb2gray(image)
    # Compute partial derivatives
    I_x2, I_y2, I_xy = Sobel_filtering(image)

    # Compute second moment matrix in a Gaussian window around each pixel
    I_x2 = gaussian_filtering(I_x2, 3, 1)
    I_y2 = gaussian_filtering(I_y2, 3, 1)
    I_xy = gaussian_filtering(I_xy, 3, 1)

    #  Compute corner response function and get the location of corner
    corner_list = corner_detect(I_x2, I_y2, I_xy,alpha,threshold)

    return corner_list


if __name__ == '__main__':
    image = data.checkerboard()

    # My Harris corner detection
    corner_list = my_harris_corner_detector(image,0.04,0.05)
    # Harris corner detection
    harris_response = corner_harris(image)
    coordinates_peaks = corner_peaks(harris_response, min_distance=5, threshold_rel=0.05)
    # the number of overlapped corners detected by the two methods
    overlapped = [i for i in coordinates_peaks if i in corner_list]
    fig, axarr = plt.subplots(1, 3, figsize=(20, 5))
    axarr[0].imshow(image, cmap="gray")
    axarr[0].set_title('Gray scale image')
    axarr[0].axis('off')
    axarr[1].imshow(image, cmap="gray")
    axarr[1].plot(coordinates_peaks[:, 0], coordinates_peaks[:, 1], '+b', markersize=15)
    axarr[1].set_title('Harris corner detection number = '+ str(len(coordinates_peaks)))
    axarr[1].axis('off')
    axarr[2].imshow(image, cmap="gray")
    axarr[2].scatter(corner_list[:, 0], corner_list[:, 1], s=100, facecolors='none',edgecolors='r')
    axarr[2].set_title('My harris corner detection number = '+ str(len(corner_list)))
    axarr[2].axis('off')
    fig.text(0,0,"the number of overlapped corners detected by the two methods : " + str(len(overlapped)), horizontalalignment='left',
     verticalalignment='bottom',fontsize=20)
    plt.show()
