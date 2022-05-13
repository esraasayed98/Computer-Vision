from PyQt5.QtCore import right
import numpy as np
from numpy.core.defchararray import array
from optimal_threashold import optimal_thresholding
from otsu import Otsu_threshold
from spectral import spectral_th




def local(image, method):
    out_image = image
    img_1=[]
    img_2=[]
    img_3=[]
    img_4=[]
    
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0

    th1, th2 = 0,0
    th3, th4 = 0,0
    th5, th6 = 0,0
    th7, th8 = 0,0
    imgs_arr=[np.array(img_1), np.array(img_2), np.array(img_3), np.array(img_4)]
    
    for r in range (0,int(image.shape[0]/2)):
        left_upper=[]
        right_upper=[]
        for k in range (0,int(image.shape[1]/2)):
                left_upper.append(image[r][k])
                
        img_1.append(left_upper)
        for u in range (int(image.shape[1]/2), image.shape[1]):
                
                right_upper.append(image[r][u])
        img_2.append(right_upper)
    
    for r in range (int(image.shape[0]/2), image.shape[0]):
        left_lower=[]
        right_lower=[]
        for k in range (0,int(image.shape[1]/2)):
                left_lower.append(image[r][k])
                
        img_3.append(left_lower)
        for u in range (int(image.shape[1]/2), image.shape[1]):
                right_lower.append(image[r][u])
                
        img_4.append(right_lower)
    
    if method == "Otsu":

            
            t1 = Otsu_threshold(np.array(img_1))
            t2 = Otsu_threshold(np.array(img_2))
            t3 = Otsu_threshold(np.array(img_3))
            t4 = Otsu_threshold(np.array(img_4))



    elif method == "optimal":

            t1 = optimal_thresholding(np.array(img_1))
            t2 = optimal_thresholding(np.array(img_2))
            t3 = optimal_thresholding(np.array(img_3))
            t4 = optimal_thresholding(np.array(img_4))

    elif method == "spectral":
            th1, th2 = spectral_th(img_1)
            th3, th4 = spectral_th(img_2)
            th5, th6 = spectral_th(img_3)
            th7, th8 = spectral_th(img_4)
            
    else:
        print("invaild method")

    max_pixel_img1=(np.array(img_1)).max()
    max_pixel_img2=(np.array(img_2)).max()
    max_pixel_img3=(np.array(img_3)).max()
    max_pixel_img4=(np.array(img_4)).max()
    max_img_px = [max_pixel_img1, max_pixel_img2, max_pixel_img3, max_pixel_img4]
    if (method == "optimal") or (method == "Otsu"):
   
        
        for r in range (0,int(image.shape[0]/2)):
            for k in range (0,int(image.shape[1]/2)):
                out_image[r][k] = ((out_image[r][k]) >= t1)*max_pixel_img1
                
            for u in range (int(image.shape[1]/2), image.shape[1]):
                out_image[r][u] = (out_image[r][u] >= t2)*max_pixel_img2

        for r in range (int(image.shape[0]/2), image.shape[0]):
            for k in range (0,int(image.shape[1]/2)):
                out_image[r][k] = (out_image[r][k] >= t3)*max_pixel_img3
            for u in range (int(image.shape[1]/2), image.shape[1]):
                out_image[r][u] = (out_image[r][u] >= t4)*max_pixel_img4

    elif method == "spectral":
        
        for r in range (0,int(image.shape[0]/2)):
            
            for k in range (0,int(image.shape[1]/2)):
                
                pixel = image[r][k]
                if pixel <= th1:
                    out_image[r][k] = 0
                elif pixel > (th2):
                    out_image[r][k]=max_pixel_img1
                else:
                    out_image[r][k]=(max_pixel_img1/2)
         
            for u in range (int(image.shape[1]/2), image.shape[1]):

                pixel = image[r][u]
                if pixel <= th3:
                    out_image[r][u] = 0
                elif pixel > (th4):
                    out_image[r][u]=max_pixel_img2
                else:
                    out_image[r][u]=(max_pixel_img2/2)
            

        for r in range (int(image.shape[0]/2), image.shape[0]):
            
            for k in range (0,int(image.shape[1]/2)):
                pixel = image[r][k]
                if pixel <= th5:
                    out_image[r][k] = 0
                elif pixel > (th6):
                    out_image[r][k]=max_pixel_img3
                else:
                    out_image[r][k]=(max_pixel_img3/2)
            
            for u in range (int(image.shape[1]/2), image.shape[1]):

                pixel = image[r][u]
                if pixel <= th7:
                    out_image[r][u] = 0
                elif pixel > (th8):
                    out_image[r][u]=max_pixel_img4
                else:
                    out_image[r][u]=(max_pixel_img4/2)
            
    return out_image
        