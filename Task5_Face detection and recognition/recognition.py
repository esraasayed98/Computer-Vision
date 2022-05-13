import os, glob
from sklearn import preprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import pathlib

def recognition(image):
    
    Faces_test_data = os.getcwd()+'/test_cropped/' + image
    Faces_data = os.getcwd()+'/cropped_faces-train/'  
    image_num =0
    names=[]
    resized_shape=(120,80)
    images_arr=[]


    for i,image in enumerate(glob.glob(Faces_data + '/*')):

            path=pathlib.PurePath(image)
            image_name=path.name
            person_id=image_name[0:3]
            names.append(person_id) 
            read_image = cv2.imread(image,0)       
            resize_image =cv2.resize(read_image, (resized_shape[1], resized_shape[0]))
            
            images_arr.append(resize_image)
            
            image_num += 1
    images_arr = np.array(images_arr)
    images_arr.resize((image_num,resized_shape[0]*resized_shape[1]))

    mean_vector = np.sum(images_arr,axis=0,dtype='float64')/image_num
    mean_matrix = np.tile(mean_vector,(image_num,1))
    #print(mean_matrix.shape)
    normalize_faces = images_arr - mean_matrix

    Cov_mat_dim_redu = (normalize_faces.dot(normalize_faces.T))/image_num
    #print(Cov_mat_dim_redu.shape)
    eig_vals,eig_vects = np.linalg.eig(Cov_mat_dim_redu)

    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vects  = eig_vects [:,idx]
    back_original_dim = normalize_faces.T @ eig_vects
    eigenfaces = preprocessing.normalize(back_original_dim.T)
    #print(eigenfaces.shape)

    # Testing on the image
    test_img = cv2.imread(Faces_test_data, cv2.IMREAD_GRAYSCALE)
    print(Faces_test_data)
    test_img = cv2.resize(test_img,(resized_shape[1],resized_shape[0]))
    mean_sub_testimg=np.reshape(test_img,(test_img.shape[0]*test_img.shape[1])) - mean_vector
    select_k_vectors=450
    E = eigenfaces[:select_k_vectors].dot(mean_sub_testimg)
    threshold = 3000
    small_distance =None 
    subject_id = None 
    for i in range(image_num):
        E_i=eigenfaces[:select_k_vectors].dot(normalize_faces[i])
        diff = E-E_i
        epsilon_i = math.sqrt(diff.dot(diff))
        if small_distance==None:
            small_distance=epsilon_i
            subject_id=i
        if small_distance >= epsilon_i:
            small_distance = epsilon_i
            subject_id=i
        
    if small_distance<threshold:
        return names[subject_id]
    else:
        return"Unknown Face"