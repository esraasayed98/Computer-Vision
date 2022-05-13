
import numpy as np
def Otsu_threshold(img):
    # for normalozing the entered image to be with pixels in range of (0,255)
    im = ((img - img.min()) * (1/(img.max() - img.min())) * 255).astype('uint8')
    # now we get the histogram
    histogram = [np.sum(im == i) for i in range(256)]

    """initializing an object to store the threshold value and the variance value 
    to reach the maximum variance and its corresponding threshold """
    sigma_b_max = (0, -np.inf)

    for threshold in range(256):
        # the number of pixels in region below the threshold
        n1 = sum(histogram[:threshold])
        # the number of pixels in region upper the threshold
        n2 = sum(histogram[threshold:])
        # calculating the mean for region below the threshold
        mean1 = sum([i * histogram[i] for i in range(0 , threshold)]) / n1 if n1 > 0 else 0
        # calculating the mean for region upper the threshold
        mean2 = sum([i * histogram[i] for i in range(threshold ,0)]) / n2 if n2 > 0 else 0

        # calculate the between_variance(sigma_b)
        sigma_b = n1 * n2 *(mean1 - mean2) ** 2
        #check if the variance in sigma_b is greater than sigma_b_max
        if sigma_b > sigma_b_max[1]:
            """if true ,it updates the old values in sigma_b_max to keep only the maximum variance
            and the corresponding threshold after iterating"""
            sigma_b_max = (threshold , sigma_b)

    """after the loop terminates, we get the final threshold at which the between_variance is maximized 
    and map it to the image dimensions"""

    otsu_th = (sigma_b_max[0] / 255) * (img.max() - img.min()) + img.min()
    

    return otsu_th


