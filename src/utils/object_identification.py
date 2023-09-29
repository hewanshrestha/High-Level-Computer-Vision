from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from .image_filtering import rgb2gray
from .image_histograms import is_grayvalue_hist, get_dist_by_name, get_hist_by_name


def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    # Your code here
    for i in range(0, len(model_hists)):
        for j in range(0, len(query_hists)):
            D[i,j] = get_dist_by_name(model_hists[i], query_hists[j], dist_type)
    
    # raise NotImplementedError
    best_match = []
    for i in range(0, D.shape[1]):
        best_match.append(np.argmin(D[:, i]))

    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):

    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist
    # Your code here
    
    if hist_isgray:
        for i in range(0, len(image_list)): 
            image_hist.append(get_hist_by_name(rgb2gray(np.array(Image.open(image_list[i])).astype('double')), num_bins, hist_type))
            
    if hist_isgray == False:
        for i in range(0, len(image_list)): 
            image_hist.append(get_hist_by_name(np.array(Image.open(image_list[i])).astype('double'), num_bins, hist_type))

    # raise NotImplementedError

    return np.array(image_hist)


def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()
    num_nearest = 5  # Show the top-5 neighbors
    
    # Your code here
    
    [_, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    for i in range(0, len(query_images)): 
        N = np.argsort(D, axis=0)[:5,:][:,i]
        plt.figure()
        plt.subplot(1,6,1); plt.imshow(np.array(Image.open(query_images[i])), vmin=0, vmax=255); plt.title("Query Image")
        plt.subplot(1,6,2); plt.imshow(np.array(Image.open(model_images[N[0]])), vmin=0, vmax=255); plt.title("N1")
        plt.subplot(1,6,3); plt.imshow(np.array(Image.open(model_images[N[1]])), vmin=0, vmax=255); plt.title("N2")
        plt.subplot(1,6,4); plt.imshow(np.array(Image.open(model_images[N[2]])), vmin=0, vmax=255); plt.title("N3")
        plt.subplot(1,6,5); plt.imshow(np.array(Image.open(model_images[N[3]])), vmin=0, vmax=255); plt.title("N4")
        plt.subplot(1,6,6); plt.imshow(np.array(Image.open(model_images[N[4]])), vmin=0, vmax=255); plt.title("N5")
        plt.show()

    # raise NotImplementedError


