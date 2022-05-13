import numpy as np
import math
import cv2
from collections import defaultdict
from math import sqrt, pi, cos, sin
class Functions():

    def SaltAndPepperNoise(self, img, percetage):
        SP_NoiseImg = img
        SP_NoiseNum = int(percetage * img.shape[0] * img.shape[1])
        for i in range(SP_NoiseNum):
            # randX->represents a randomly generated row
            # randY->represents a randomly generated column
            # random.randint()->generate random integers

            randX = np.random.randint(0, img.shape[0] - 1)
            randY = np.random.randint(0, img.shape[1] - 1)
            if np.random.randint(0, 1) == 0:
                SP_NoiseImg[randX, randY] = 255  # 1 is salt particle noise

            else:
                SP_NoiseImg[randX, randY] = 0  # 0 is pepper noise
        return SP_NoiseImg

    def addGaussianNoise(self, image, mean=0, var=0.001):
        # Add Gaussian noise
        # mean:mean
        # var:variance

        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        noise_img = image + noise
        if noise_img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        noise_img = np.clip(noise_img, low_clip, 1.0)
        noise_img = np.uint8(noise_img * 255)

        return noise_img

    def uniform_noise(self,image):
        uniform_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.randu(uniform_noise, 0, 255)
        uniform_noise = (uniform_noise * 0.5).astype(np.uint8)
        noisy_image2 = cv2.add(image, uniform_noise)
        return noisy_image2


    def Avg_Filter(self,img):

        # Obtain number of rows and columns of the image
        m, n = img.shape
        # Develop Averaging filter(3, 3) mask
        mask = np.ones([3, 3], dtype=int)
        mask = mask / 9

        # Convolve the 3X3 mask over the image
        img_new = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2] + \
                       img[
                           i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[
                           i + 1, j - 1] * mask[
                           2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

                img_new[i, j] = temp
        img_new = img_new.astype(np.uint8)
        return img_new

    def Med_Filter(self,img_noisy):
        # Obtain the number of rows and columns
        # of the image
        m, n = img_noisy.shape

        # Traverse the image. For every 3X3 area,
        # find the median of the pixels and
        # replace the center pixel by the median
        filtered_img = np.zeros([m, n])

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [img_noisy[i - 1, j - 1],
                        img_noisy[i - 1, j],
                        img_noisy[i - 1, j + 1],
                        img_noisy[i, j - 1],
                        img_noisy[i, j],
                        img_noisy[i, j + 1],
                        img_noisy[i + 1, j - 1],
                        img_noisy[i + 1, j],
                        img_noisy[i + 1, j + 1]]

                temp = sorted(temp)
                filtered_img[i, j] = temp[4]

        filtered_img = filtered_img.astype(np.uint8)
        return filtered_img

    def convolution(self,image, kernel, average=False):

        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape

        output = np.zeros(image.shape)

        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)

        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]

        return output



    def dnorm(self,x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def gaussian_kernel(self,size, sigma=1):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

        kernel_2D *= 1.0 / kernel_2D.max()

        return kernel_2D

    def gaussian_blur(self,image, kernel_size):
        kernel = self.gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
        g_img = self.convolution(image, kernel, average=True)
        return g_img






    def sobel(self , img):
        sobelX = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]], dtype="int")
        # construct the Sobel y-axis kernel
        sobelY = np.array((
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]), dtype="int")
        sobelX_edges = self.convolution(img, kernel=sobelX / 8)
        sobelY_edges = self.convolution(img, kernel=sobelY / 8)

        sobel_total = np.sqrt(sobelY_edges ** 2 + sobelX_edges ** 2)
        Theta = np.arctan2(sobelY_edges, sobelX_edges) * 180 / np.pi
        return sobel_total,Theta

    def prewitt(self,img):
        PrewittX = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]], dtype="int")

        PrewittY = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]], dtype="int")
        PrewittX_edges = self.convolution(img, kernel=PrewittX / 6)
        PrewittY_edges = self.convolution(img, kernel=PrewittY / 6)
        prewitt_total = np.sqrt(PrewittY_edges ** 2 + PrewittX_edges ** 2)
        return prewitt_total


    def Roberts(self, img):
        RobertX = np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 0, 1]], dtype="int")

        RobertY = np.array([
            [0, 0, 0],
            [0, 0, -1],
            [0, 1, 0]], dtype="int")
        RobertX_edges = self.convolution(img, kernel=RobertX)
        RobertY_edges = self.convolution(img, kernel=RobertY)
        Roberts_total = np.sqrt(RobertY_edges ** 2 + RobertX_edges ** 2)
        return Roberts_total

    def canny(self,img):
        f_img = self.gaussian_blur(img, 5)
        sobel_grad , sobel_direction=self.sobel(f_img)
        non_max=self.maximum(sobel_grad,sobel_direction)
        out=self.thresholding(non_max)
        return out


    def maximum(self, det, phase):
        gmax = np.zeros(det.shape)
        for i in range(gmax.shape[0]):
            for j in range(gmax.shape[1]):
                if phase[i][j] < 0:
                    phase[i][j] += 360

                if ((j + 1) < gmax.shape[1]) and ((j - 1) >= 0) and ((i + 1) < gmax.shape[0]) and ((i - 1) >= 0):
                    # 0 degrees
                    if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                        if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                            gmax[i][j] = det[i][j]
                    # 45 degrees
                    if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                        if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                            gmax[i][j] = det[i][j]
                    # 90 degrees
                    if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                        if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                            gmax[i][j] = det[i][j]
                    # 135 degrees
                    if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                        if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                            gmax[i][j] = det[i][j]
        return gmax

    def thresholding(self,im):
        thres = np.zeros(im.shape)
        mmax = np.amax(im)
        lo, hi = 0.1 * mmax, 0.2 * mmax

        # If edge intensity is greater than 'High' it is a sure-edge
        # below 'low' threshold, it is a sure non-edge
        strong_i, strong_j = np.where(im >= hi)
        zeros_i, zeros_j = np.where(im < lo)

        # weak edges
        weak_i, weak_j = np.where((im <= hi) & (im >= lo))

        # Set same intensity value for all edge pixels
        thres[strong_i, strong_j] = 255
        thres[zeros_i, zeros_j] = 0
        thres[weak_i, weak_j] = 75

        M, N = thres.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (thres[i, j] == 75):
                    if 255 in [thres[i + 1, j - 1], thres[i + 1, j], thres[i + 1, j + 1], thres[i, j - 1],
                               thres[i, j + 1], thres[i - 1, j - 1], thres[i - 1, j], thres[i - 1, j + 1]]:
                        thres[i, j] = 255
                    else:
                        thres[i, j] = 0
        return (thres)

    def fourier(self,img):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        self.dft_shift = np.fft.fftshift(dft)  ### Fourier Response
        magnitude_spectrum = 20 * np.log(
            (cv2.magnitude(self.dft_shift[:, :, 0], self.dft_shift[:, :, 1])))  # channel 0 is real && channel 1 is imaginary
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        return rows, cols,crow, ccol
    def high_pass_filter(self,img):
        rows, cols, crow, ccol=self.fourier(img)
        ### HIGH PASS FILTER
        mask_high = np.ones((rows, cols, 2), np.int8)
        r = 5
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        maskHigh_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask_high[maskHigh_area] = 0
        fshift_high = self.dft_shift * mask_high
        fshift_mask_mag_high = 20 * np.log(cv2.magnitude(fshift_high[:, :, 0], fshift_high[:, :, 1]))
        f_ishift_high = np.fft.ifftshift(fshift_high)
        img_back_high = cv2.idft(f_ishift_high)
        img_back_high = cv2.magnitude(img_back_high[:, :, 0], img_back_high[:, :, 1])

        return img_back_high

    def low_pass_filter(self,img):
        rows, cols, crow, ccol = self.fourier(img)
        ### LOW PASS FILTER
        mask_low = np.zeros((rows, cols, 2), np.int8)
        r = 23
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        maskLow_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask_low[maskLow_area] = 1
        fshift_low = self.dft_shift * mask_low
        fshift_mask_mag_low = 20 * np.log(cv2.magnitude(fshift_low[:, :, 0], fshift_low[:, :, 1]))
        f_ishift_low = np.fft.ifftshift(fshift_low)
        img_back_low = cv2.idft(f_ishift_low)
        img_back_low = cv2.magnitude(img_back_low[:, :, 0], img_back_low[:, :, 1])
        return img_back_low

    def Histogram_Computation_color(self, Image):
        Image_Height = Image.shape[0]
        Image_Width = Image.shape[1]
        Image_Channels = Image.shape[2]
        Histogram = np.zeros([256, Image_Channels], np.int32)
        for x in range(0, Image_Height):
            for y in range(0, Image_Width):
                for c in range(0, Image_Channels):
                    Histogram[Image[x, y, c], c] += 1
        return Histogram

    def Equalization(self, img_flatten,shape):
        # Normalizing the histogram
        histogram = np.bincount(img_flatten)
        num_pixels = np.sum(histogram)
        normalized_histogram = histogram / num_pixels
        # Normalized cummulative histogram
        cummulative_histogram = np.cumsum(normalized_histogram)
        # Transforming map
        trans_map = np.floor(255 * cummulative_histogram).astype(np.uint8)
        # pixel_mapping
        equalized_image = []
        for i in list(img_flatten):
            equalized_image.append(trans_map[i])
        # Reshaping and writing the new image
        equalized_image_array = np.reshape(np.asarray(equalized_image), shape)
        return equalized_image_array
    
    
    #hough transform
    ## Voting
    def voting(self,edge_img):
        h, w = edge_img.shape
        # get rho max length
        rho_max = np.ceil(np.sqrt(h ** 2 + w ** 2)).astype(np.int)
        # hough accumulator
        hough = np.zeros((rho_max, 180), dtype=np.int)
        # get location of edge  
        coordinates = np.where(edge_img == 255)
     
        ## hough transformation
        for y, x in zip(coordinates[0], coordinates[1]):
                for theta in range(0, 180, 1):
                        # get polar coordinates
                        t = np.pi / 180 * theta
                        rho = int(x * np.cos(t) + y * np.sin(t))
                        # vote
                        hough[rho, theta] += 1
       
        return hough
    
    # non maximum suppression
    def non_maximum_suppression(self,hough):
        rho_max, _ = hough.shape
    
        ## non maximum suppression 
        for rho in range(rho_max):
            for theta in range(180):
                # get 8 nearest neighbor
                theta1 = max(theta-1, 0)
                theta2 = min(theta+2, 180)
                rho1 = max(rho-1, 0)
                rho2 = min(rho+2, rho_max-1)
                if np.max(hough[rho1:rho2, theta1:theta2]) == hough[rho,theta] and hough[rho, theta] != 0:
                    pass
                else:
                    hough[rho,theta] = 0
     
        return hough
    
    def inverse_hough(self,hough, img):
        h, w,_= img.shape
        rho_max, _ = hough.shape
         
        out_img = img.copy()
        x_cord = np.argsort(hough.ravel())[::-1][:24]
        y_cord = x_cord.copy()
        thetas = x_cord % 180
        rhos = y_cord // 180
         
        # for each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta
         
            #convert from hough space to image space (x,y)
            for x in range(w):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= h or y < 0:
                        continue
                    out_img[y, x] = (139,69,19)
            for y in range(h):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= w or x < 0:
                        continue
                    out_img[y, x] = (0,0,255)
                
        out_img = out_img.astype(np.uint8)
         
        return out_img

    def Hough_Circle(self,org_image,edges_detected):
        # Find circles

        # rgb_img = cv2.cvtColor(org_image, cv2.COLOR_GRAY2RGB)
        out_img = org_image.copy()

        rmin = 10
        rmax = 40
        steps = 100 ## when trying 180 to include all angles, not all circles are detected
        threshold = 0.3

        points = []
        for r in range(rmin, rmax + 1):
            for t in range(steps):
                points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

        acc = defaultdict(int)
        for x in range(edges_detected.shape[1]):
            for y in range(edges_detected.shape[0]):
                if edges_detected[y][x] != 0:
                    for r, dx, dy in points:
                        a = x - dx
                        b = y - dy
                        acc[(a, b, r)] += 1

        circles = []
        ## sorted function  --->  if a key has more than one value, only the maximum is kept (the value here is the accumulator count)
        for k, v in sorted(acc.items()): 
            x, y, r = k
            if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > 1.3 *(rc ** 2) for xc, yc, rc in circles):
                # the condition above to exclude the circles that are very near to each other, and make sure that at least 30% of the circle is there
                print(v / steps, x, y, r)
                circles.append((x, y, r))

        for x, y, r in circles:
            cv2.circle(out_img, (x,y), r, (0,255,0), 2)

        return out_img            