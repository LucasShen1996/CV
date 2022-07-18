from skimage.color import *
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import canny
from skimage.filters import gaussian


def create_collage(address_of_folder: str):
    image_list = []
    for filename in os.listdir(address_of_folder):
        # read file save in list
        image_list.append(plt.imread(address_of_folder + '/' + filename))
    height, width, depth = image_list[0].shape
    # change to gray image
    gray_image_list = []
    for i in range(len(image_list)):
        gray_image_list.append(rgb2gray(image_list[i]))
    # get each files histogram list
    image_hist_list = []
    for i in range(len(gray_image_list)):
        image_hist_list.append(np.histogram(gray_image_list[i].ravel(), 256)[0])
    # get the standard deviation for each image's histogram
    color_std_hist = []
    for i in range(len(image_hist_list)):
        color_std_hist.append(np.std(image_hist_list[i]))
    # sort image by the standard deviation for each image's histogram (color)
    color_sort_index = np.argsort(color_std_hist)[::-1]
    color_sort_index = color_sort_index.tolist()
    # find each image's corner's number
    edge_map_list = []
    for i in range(len(gray_image_list)):
        edge_map_list.append(canny(gray_image_list[i], sigma=3))
    edge_hist_list = []
    for i in range(len(edge_map_list)):
        edge_hist_list.append(np.histogram(edge_map_list[i].ravel(), 256)[0][-1])
    # sort image by corner's number
    edge_sort_index = np.argsort(edge_hist_list)[::-1]
    edge_sort_index = edge_sort_index.tolist()

    print(color_sort_index)
    print(edge_sort_index)
    # do the mountain sort
    mountain_sort_color = color_sort_index[::2]
    mountain_sort_color.extend(color_sort_index[1::2][::-1])
    mountain_sort_edge = edge_sort_index[::2]
    mountain_sort_edge.extend(edge_sort_index[1::2][::-1])
    # resort the image list
    resort_image_list_color = []
    resort_image_list_corner = []
    for i in range(len(mountain_sort_color)):
        resort_image_list_color.append(image_list[mountain_sort_color[i]])
        resort_image_list_corner.append(image_list[mountain_sort_edge[i]])

    # overlapping function
    def overlapping_image(alpha, image1, beta, image2, size=200, sigma=2):
        overlapping_part = (alpha * image1[:, -size:image1.shape[1], :] + beta * image2[:, 0:size, :])
        # do the gaussian filter
        overlapping_part = gaussian(overlapping_part, sigma, multichannel=2)
        # overlapping_part = gaussian(overlapping_part, sigma, channel_axis=2)
        return overlapping_part

    # overlapping
    overlapping_part = []
    for i in range(1, len(resort_image_list_color)):
        overlapping_part.append(overlapping_image(alpha=0.5, image1=resort_image_list_color[i - 1], beta=0.5,
                                                  image2=resort_image_list_color[i], size=200, sigma=3))
    # rechange the image with the overlapping part
    image_rechange = []
    image_rechange.append(resort_image_list_color[0][:, 0:-200, :])
    temp = np.zeros([height, width - 200, depth])
    for i in range(1, len(resort_image_list_color)):
        for w in range(200):
            resort_image_list_color[i][:, w, :] = overlapping_part[i - 1][:, w, :]
        if i == 4:
            image_rechange.append(resort_image_list_color[i][:, :, :])
        else:
            image_rechange.append(resort_image_list_color[i][:, 0:-200, :])

    re_height, re_width, re_depth = image_rechange[0].shape
    output_image = np.zeros([height, re_width * len(image_list) + 200, depth])
    # connect images
    for i in range(len(image_list)):
        if i == 4:
            for col in range(0, width):
                output_image[:, col + (i * re_width), :] = image_rechange[i][:, col, :] / 255
        else:
            for col in range(0, re_width):
                output_image[:, col + (i * re_width), :] = image_rechange[i][:, col, :] / 255
    #  plot image
    fig, axes = plt.subplots(1, ncols=1, figsize=(100, 100))
    axes.imshow(output_image)
    axes.axis("off")
    plt.show()


if __name__ == '__main__':
    path = "./task1"
    create_collage(path)

