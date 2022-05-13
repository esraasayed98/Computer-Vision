import cv2, math
import numpy as np
import matplotlib.pyplot as plt
import time 

def convolution(image, kernel, average=False):
    
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

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D

def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    g_img = convolution(image, kernel, average=True)
    return g_img


def sobel(img):
    sobelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]], dtype="int")
    # construct the Sobel y-axis kernel
    sobelY = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]), dtype="int")
    sobelX_edges = convolution(img, kernel=sobelX / 8)
    sobelY_edges = convolution(img, kernel=sobelY / 8)

    sobel_total = np.sqrt(sobelY_edges ** 2 + sobelX_edges ** 2)
    Theta = np.arctan2(sobelY_edges, sobelX_edges) * 180 / np.pi
    return sobelX_edges,sobelY_edges


def get_covariance_matrix(image):
  sobelX, sobelY = sobel(image)

  Ixx = sobelX * sobelX
  Ixy = sobelX * sobelY
  Iyx = sobelY * sobelX
  Iyy = sobelY * sobelY

  return Ixx, Ixy, Iyx, Iyy

def get_all_M_matrices(image):
  Ixx, Ixy, Iyx, Iyy = get_covariance_matrix(image)
  windowed_Ixx = gaussian_blur(Ixx, 7) #apply_gaussian(Ixx, 7, 1.5)
  windowed_Ixy = gaussian_blur(Ixy, 7) #apply_gaussian(Ixy, 7, 1.5)
  windowed_Iyx = gaussian_blur(Iyx, 7) #apply_gaussian(Iyx, 7, 1.5)
  windowed_Iyy = gaussian_blur(Iyy, 7) #apply_gaussian(Iyy, 7, 1.5)

  return windowed_Ixx, windowed_Ixy, windowed_Iyx, windowed_Iyy

def get_responses(image):
  windowed_Ixx, windowed_Ixy, windowed_Iyx, windowed_Iyy = get_all_M_matrices(image)
  k = 0.04
  responses = np.zeros((image.shape[0], image.shape[1]))
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      first = windowed_Ixx[i][j]
      second = windowed_Ixy[i][j]
      third = windowed_Iyx[i][j]
      fourth = windowed_Iyy[i][j]

      M_matrix_i_j = np.array([[first, second],[third, fourth]])
      determinant = np.linalg.det(M_matrix_i_j)
      trace = np.matrix.trace(M_matrix_i_j)
      response_i_j = determinant - (k*(trace**2))
      responses[i][j] = response_i_j

  return responses

def get_max_indices(block):
  m = block.shape[0]
  n = block.shape[1]

  max_val = np.amin(block) - 1
  p = -1
  q = -1

  for i in range(m):
    for j in range(n):
      if block[i][j] > max_val:
        max_val = block[i][j]
        p = i
        q = j
  return p, q, max_val

def non_maxima_suppression(responses, window_size):
  all_y = []
  all_x = []

  height = responses.shape[0]
  width = responses.shape[1]

  count = 0
  for i in range(0,width-window_size+1,window_size):
    for j in range(0,height-window_size+1,window_size):
      current_block = responses[j:j+window_size, i:i+window_size]
      p, q, max_val = get_max_indices(current_block)
      if(max_val!=-1):
        global_row = j + p
        global_col = i + q
        all_y.append(global_row)
        all_x.append(global_col)
  
  return all_y, all_x


def harris(grayscale_image):
    harris_start=time.time()

    threshold = 10000

    responses = get_responses(grayscale_image)
    responses = np.where(responses>threshold,responses,-1)
    ################################
    rows, cols = non_maxima_suppression(responses, 13)
    

    new_features = np.zeros(grayscale_image.shape)

    for each_corner in range(len(rows)):
        cv2.circle(grayscale_image, (cols[each_corner], rows[each_corner]), 1, (0,0,255), -1)
        cv2.circle(new_features, (cols[each_corner], rows[each_corner]),1,(255,255,255), -1)

    harris_end=time.time()
    harris_time=harris_end - harris_start
    return harris_time