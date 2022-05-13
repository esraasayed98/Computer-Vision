from PyQt5 import QtWidgets, uic,QtCore, QtGui
from PyQt5.QtGui import * 
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog
import imageio
import os.path
import sys
import cv2
from functions import Functions
import numpy as np
import pyqtgraph as pg
from PIL import Image , ImageDraw
import skimage.filters
from skimage.filters import threshold_local
from snake import Snake
from padding import padd_with_first_col_row
import matplotlib.pyplot as plt
import time
import pathlib

import sift
from matcher import siftmatcher
from harris import harris
from optimal_threashold import optimal_thresholding
from otsu import Otsu_threshold
from spectral import spectral_th
from kmean_functions import kmean
from region_growing import regionGrow
from mean_shift import mean_shift
from Agglomerative2 import Agglomerative
from local_thresholding import local
from face_detection import face_detection
from recognition import recognition 

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        uic.loadUi('ui.ui', self)
        self.fun=Functions()
        self.box = [self.Add_noise_Box,self.Filters_Box , self.Edge_Box]
        self.functions=[self.Add_noise, self.filter ,self.Detect_edges ]

        for i in range(len(self.box)):
            self.box[i].setEnabled(False)
            self.box[i].activated.connect(self.functions[i])
        # image view widgets

        self.widgets=[self.output_widget, self.filtered_widget,self.edge_widget,
                      self.freq_filter_widget,self.hybird_widget, self.output_img]
        for i in range(len(self.widgets)):
            self.widgets[i].ui.histogram.hide()
            self.widgets[i].ui.roiBtn.hide()
            self.widgets[i].ui.menuBtn.hide()
            self.widgets[i].ui.roiPlot.hide()

        # connect Buttons
        self.Load_Img_Button.clicked.connect(lambda: self.Add_Img(1))
        self.Load_Img_Button_2.clicked.connect(lambda: self.Add_Img(2))
        self.Freq_filter_Box.activated.connect(self.freq_filters)
        #hybrid
        self.hyb_img1_button.clicked.connect(lambda: self.Add_Img(3))
        self.hyb_img2_button.clicked.connect(lambda: self.Add_Img(4))
        self.hybird_Button.setEnabled(False)
        self.hybird_Button.clicked.connect(self.hybrid)

        #Histogram tab
        self.check = 0
        self.combo_options = ["Equalization", "Normalization", "Global Threshold", "Local Threshold", "Color To Gray"]
        self.histo = [self.input_hist, self.output_hist]
        self.image = 0
        self.load.clicked.connect(lambda: self.add_image())
        self.comboBox.currentIndexChanged.connect(lambda: self.add_hist_in(self.image))
        self.comboBox.setEnabled(False)

        self.H1 = False
        self.H2 = False       
        #Hough transform tab
        self.hough_load_button.clicked.connect(lambda: self.Add_Img(5))
        self.detect_lines_Button.clicked.connect(self.detect_lines)     
        self.detect_circles_button.clicked.connect(self.detect_circles)
        #Snake tab
        self.circle.setEnabled(False)
        self.snake.setEnabled(False)
        self.fin.setText("...")
        self.path=0
        self.image_shape=0
        self.init=[]
        self.x=0
        self.y=0
        self.r=0
        self.iterator=0
        self.h=0
        self.gif_time=0
        self.image_s.clicked.connect(lambda: self.draw())
        self.snake.clicked.connect(lambda: self.result())
        self.circle.clicked.connect(lambda: self.Draw_circle())
        self.fin.setStyleSheet("color: red")
        self.movie = QMovie('result.gif')
        self.movie.finished.connect(lambda:self.add_iter())

        #feature matching
        self.labels=[self.img_mat1,self.img_mat2]
        self.image_mat=[0,0]
        self.load_img_mat1.clicked.connect(lambda: self.load_image_matching(0))
        self.load_img_mat2.clicked.connect(lambda: self.load_image_matching(1))
        self.result_mat.clicked.connect(lambda: self.matching())
        self.keypoints_set = [0,0]
        self.descriptors_set = [0,0]
        self.harris.setStyleSheet("color: red")
        self.harris.setText("...")
        self.sift.setStyleSheet("color: red")
        self.sift.setText("...")
        self.ssd.setStyleSheet("color: red")
        self.ssd.setText("...")
        #segmentation 
        self.seg_img_load.clicked.connect(self.load_seg_img)
        
        
        self.enable_disable_radio(False)
        self.segmentation_img = 0
        self.radio_h_c=[self.H_based, self.C_based]
        self.radio_G_l=[self.rad_g,self.rad_l]
        self.buttongroup_G_l=QtWidgets.QButtonGroup()
        self.buttongroup_G_l.addButton(self.radio_G_l[0],0)
        self.buttongroup_G_l.addButton(self.radio_G_l[1],1)

        self.buttongroup_h_c=QtWidgets.QButtonGroup()
        self.buttongroup_h_c.addButton(self.radio_h_c[0],0)
        self.buttongroup_h_c.addButton(self.radio_h_c[1],1)
        self.arr_group_radio=[self.buttongroup_h_c, self.buttongroup_G_l]
        self.arr_arr_radio=[self.radio_h_c ,self.radio_G_l]
        self.rad_g.setEnabled(False)
        self.rad_l.setEnabled(False)
        self.optimal.clicked.connect(lambda: self.optimal_otsu_th(self.segmentation_img,0))  
        self.otsu.clicked.connect(lambda: self.optimal_otsu_th(self.segmentation_img,1))  
        self.spectral.clicked.connect(lambda: self.optimal_otsu_th(self.segmentation_img,2))  
        self.kmean.clicked.connect(lambda: self.clustring_seg(self.segmentation_img,0))  
        self.region_g.clicked.connect(lambda: self.clustring_seg(self.segmentation_img,1))  
        self.agglomerative.clicked.connect(lambda: self.clustring_seg(self.segmentation_img,2))  
        self.mean_sh.clicked.connect(lambda: self.clustring_seg(self.segmentation_img,3))  
        self.seg_label.setStyleSheet("color: red")
        self.seg_label.setText("...")
        self.H_based.toggled.connect(lambda: self.radio_checked(0))
        self.C_based.toggled.connect(lambda: self.radio_checked(1))
        
        # Face Recognition 
        self.rec_load_btn.clicked.connect(self.load_rec_img)
        self.dec_btn.clicked.connect(lambda: self.dec_face(self.read_rec_img))
        self.rec_btn.clicked.connect(lambda: self.rec_face(self.img_path_test))       
        self.dec_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self.read_rec_img =  None
        self.faces = None
        self.img_path_test =None
        self.result_btn.setStyleSheet("color: red")
        
        







    def Add_Img(self, index):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png)")
        self.image_path = file_name[0]
        self.img = cv2.imread(self.image_path, 0)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            if index==1:
                self.input_img = cv2.imread(self.image_path, 0)
                self.Img_label.setPixmap(pixmap)
                self.Add_noise_Box.setEnabled(True)
                self.Edge_Box.setEnabled(True)
            elif index==2:
                self.freq_img = cv2.imread(self.image_path, 0)
                self.freq_img_label.setPixmap(pixmap)
                self.Freq_filter_Box.setEnabled(True)

            elif index==3:
                self.H1= True
                self.hyb_img1 = cv2.imread(self.image_path, 0)
                self.img1_label.setPixmap(pixmap)


            elif index==4:
                self.H2=True
                self.hyb_img2 = cv2.imread(self.image_path, 0)
                self.img2_label.setPixmap(pixmap)

            elif index==5:
               self.hough_img = cv2.imread(self.image_path)
               self.hough_img_label.setPixmap(pixmap)
               

            if (self.H1 and self.H2):
                self.hybird_Button.setEnabled(True)
                


                



    def Add_noise(self):
        index = self.Add_noise_Box.currentIndex()
        self.Filters_Box.setEnabled(True)

        if index ==1:
            img = self.input_img
            #salt and pepper noise
            self.SP_img=self.fun.SaltAndPepperNoise(img, 0.1)
            self.output_widget.clear()
            self.output_widget.show()
            self.output_widget.setImage(self.SP_img.T)

        elif index==2:
            img = self.input_img
            self.GN_img = self.fun.addGaussianNoise(img, 0,0.01)
            self.output_widget.clear()
            self.output_widget.show()
            self.output_widget.setImage(self.GN_img.T)
        elif index==3:
            img = self.input_img
            self.Un_img = self.fun.uniform_noise(img)
            self.output_widget.clear()
            self.output_widget.show()
            self.output_widget.setImage(self.Un_img.T)


    def filter(self):
        f_index = self.Filters_Box.currentIndex()
        N_index=self.Add_noise_Box.currentIndex()

        if N_index==1:
            #SP noise
            img=self.SP_img
        elif N_index==2:
            #Gaussian noise
            img=self.GN_img
        elif N_index==3:
            img=self.Un_img
        if f_index==1:
            #Gaussian filter
            f_img = self.fun.gaussian_blur(img, 5)
        elif f_index==2:
            #Average filter
            f_img=self.fun.Avg_Filter( img)
        elif f_index==3:
            #median filter
            f_img = self.fun.Med_Filter(img)
        self.filtered_widget.show()
        self.filtered_widget.setImage(f_img.T)


    def Detect_edges(self):
        img=self.input_img
        f_img = self.fun.gaussian_blur(img, 5)
        index = self.Edge_Box.currentIndex()
        #Sobel
        if index==1:
            sobel_edges,theta=self.fun.sobel(f_img)
            self.edge_widget.show()
            self.edge_widget.setImage(sobel_edges.T)


        #prewitt
        elif index==2:
            prewitt_edges = self.fun.prewitt(f_img)
            self.edge_widget.show()
            self.edge_widget.setImage(prewitt_edges.T)
        #Canny
        elif index==3:
            canny_edges = self.fun.canny(img)
            self.edge_widget.show()
            self.edge_widget.setImage(canny_edges.T)
        #Roberts
        elif index==4:
            Roberts_edges = self.fun.Roberts(f_img)
            self.edge_widget.show()
            self.edge_widget.setImage(Roberts_edges.T)
    def freq_filters(self):
        index=self.Freq_filter_Box.currentIndex()
        if index==1:
            #high pass filter
            high_freq_img=self.fun.high_pass_filter(self.freq_img)
            self.freq_filter_widget.show()
            self.freq_filter_widget.setImage(high_freq_img.T)

        elif index==2:
            # low pass filter
            low_freq_img = self.fun.low_pass_filter(self.freq_img)
            self.freq_filter_widget.show()
            self.freq_filter_widget.setImage(low_freq_img.T)

    def hybrid(self):
        high_freq_img = self.fun.high_pass_filter(self.hyb_img1)
        low_freq_img = self.fun.low_pass_filter(self.hyb_img2)

        h1, w1 = high_freq_img.shape
        h2, w2 = low_freq_img.shape
        width=0
        height=0

        if w1<=w2:
            width = w1
        else:
            width = w2
    
        if h1<=h2:
            height = h1
        else:
            height = h2

        high_freq_img_resized = cv2.resize(high_freq_img,(height,width))
        low_freq_img_resized = cv2.resize(low_freq_img,(height,width))
        
        hybrid_img=high_freq_img_resized+low_freq_img_resized
        self.hybird_widget.show()
        self.hybird_widget.setImage(hybrid_img.T)
    #Histogram Tab
    def add_image(self):
        filename = QtWidgets.QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '':
            img = np.asarray(Image.open(path))
            self.check = np.asarray(img.shape).shape[0]
            for i in range(6):
                self.comboBox.view().setRowHidden(i, False)
            if self.check == 3:
                for i in range(1, 5):
                    self.comboBox.view().setRowHidden(i, True)
            else:
                self.comboBox.view().setRowHidden(5, True)
            self.comboBox.setEnabled(True)
            self.input_img.setPixmap(QPixmap(path))
            self.image = img
            self.add_hist_in(img)

    def add_hist_in(self, img):
        x = self.comboBox.currentIndex()
        if x == 5:
            hist_values = self.fun.Histogram_Computation_color(img)
            self.Plot_Histogram(hist_values)
        elif x == 0:
            if self.check == 3:
                hist_values = self.fun.Histogram_Computation_color(img)
                self.Plot_Histogram(hist_values)
            else:
                self.Histogram(self.flattened_img(img), 0)
        else:
            self.Histogram(self.flattened_img(img), 0)
        self.add_out(x)

    def add_out(self, x):
        if x == 5:
            red = self.image[:, :, 2]
            green = self.image[:, :, 1]
            blue = self.image[:, :, 0]
            gray = (0.299 * blue) + (0.587 * green) + (0.114 * red)

            self.output_img.show()
            self.output_img.setImage(gray.T)
            image = np.array(gray, dtype='int')
            self.Histogram(self.flattened_img(image), 1)

        elif x == 1:
            equalization = self.fun.Equalization(self.flattened_img(self.image) , self.image.shape)
            self.output_img.show()
            self.output_img.setImage(equalization.T)
            self.Histogram(self.flattened_img(equalization), 1)

        elif x == 2:
            pixels = self.image.astype('float32')
            pixels /= pixels.max()
            self.output_img.show()
            self.output_img.setImage(pixels.T)
            pixels = np.array(pixels, dtype='int')
            self.Histogram(self.flattened_img(pixels), 1)

        elif x == 3:
            th = 125
            img_bool = self.image > th
            image = img_bool * 255
            self.output_img.show()
            self.output_img.setImage(image.T)
            self.Histogram(self.flattened_img(image), 1)
        elif x == 4:
            block_size = 7
            image = threshold_local(self.image, block_size, offset=10)
            self.output_img.show()
            self.output_img.setImage(image.T)
            image = np.array(image, dtype='int')
            self.Histogram(self.flattened_img(image), 1)
        else:
            self.output_hist.clear()
            self.output_img.clear()



    def Plot_Histogram(self, Histogram):
        self.input_hist.clear()
        self.input_hist.setLabel('left', 'Intensity Freq.')
        self.input_hist.setLabel('bottom', 'Intensity level')
        self.input_hist.plot(Histogram[:, 0], pen=pg.mkPen(color=(255, 0, 0)))
        self.input_hist.plot(Histogram[:, 1], pen=pg.mkPen(color=(0, 255, 0)))
        self.input_hist.plot(Histogram[:, 2], pen=pg.mkPen(color=(0, 0, 255)))

    def flattened_img(self, img):
        img_array = np.asarray(img)
        img_flatten = img_array.flatten()
        return img_flatten

    def Histogram(self, img_flatten, i):
        histogram = np.bincount(img_flatten, minlength=img_flatten.max() + 1)
        self.histo[i].clear()
        self.histo[i].setLabel('left', 'Intensity Freq.')
        self.histo[i].setLabel('bottom', 'Intensity level')
        self.histo[i].plot(histogram, pen=pg.mkPen(color=(255, 255, 255)))

    def detect_lines(self):
        
        img=self.hough_img
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge=self.fun.canny(grayscale)
        # 1-voting
        hough = self.fun.voting(edge)
        # 2-non maximum suppression
        hough = self.fun.non_maximum_suppression(hough)
        # 3-inverse hough
        out = self.fun.inverse_hough(hough, img)
        out = out.astype(np.uint8)
        cv2.imwrite("out.jpg", out)
        pixmap = QPixmap("out.jpg")
        self.hough_img_label_2.setPixmap(pixmap)

    def detect_circles(self):

        img=self.hough_img
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        detected_edges=self.fun.canny(grayscale)
        #rgb_out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        out = self.fun.Hough_Circle(self.hough_img,detected_edges)
        #out = out.astype(np.uint8)
        cv2.imwrite("out.jpg", out)
        pixmap = QPixmap("out.jpg")
        self.hough_img_label_2.setPixmap(pixmap)     
    ##################################################################
    ########## Snake Functions #################
    def draw(self):
        filename = QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '':
            self.path=path
            image_g=((Image.open(path)).resize((300,300)))
            self.plot.setPixmap(QPixmap(path))
            self.image_arr = np.asarray(image_g)
            self.image_shape=self.image_arr.shape
            print(self.image_shape)
            input_img=Image.fromarray(self.image_arr)
            input_img.save('input_img.png')
            self.path='input_img.png'
            self.circle.setEnabled(True)
            self.x=self.image_shape[0]/2
            self.y=self.image_shape[1]/2
            self.rad.setText("110")

            

    def Draw_circle(self):
        self.fin.setText("...")
        self.r=int(self.rad.text())
        s = np.linspace(0, 2*np.pi, 400)
        R = self.x + self.r*np.sin(s)
        c = self.y + self.r*np.cos(s)
        self.init = np.array([R, c]).T
        points=self.init.tolist()
        self.draw_on_image(self.path,points)
        self.plot.setPixmap(QPixmap("img.png"))
        self.snake.setEnabled(True)
        self.iter.setText("400")

    def draw_on_image(self,path,points):
        new=Image.open(path)
        points=[tuple(x) for x in points]
        draw = ImageDraw.Draw(new)
        draw.line(points,fill="red", width=3)
        new.save("img.png")



    def result(self):
        self.fin.setText("...")
        iterations=int(self.iter.text())
        gif_images=[]
        left=self.x-self.r+1
        top=self.y-self.r+1
        right=self.x+self.r-1
        bottom=self.y+self.r-1
        img=Image.open(self.path)
        crobed_img=np.asarray(img.crop((left,top,right,bottom)).convert('L'))
        print(self.image_shape)
        crobed_img=padd_with_first_col_row(crobed_img,self.image_shape)
        croped_shape=crobed_img.shape
        image_gaussian = skimage.filters.gaussian(crobed_img, 6.0)
        snakeContour,self.iterator = Snake(image_gaussian,self.init, wLine=0, wEdge=1.0, alpha=0.1, beta=10, gamma=0.001,
                                        maxIterations=iterations, maxPixelMove=1, convergence=0.1)

            
        
        for i in range (len(snakeContour)):
            points=snakeContour[i].tolist()
            self.draw_on_image(self.path,points)
            gif_images.append(imageio.imread('img.png'))
        imageio.mimsave('result.gif', gif_images, duration=0.1,loop=1)
        self.plot.setMovie(self.movie)
            
        self.movie.start()

    def add_iter(self):
        self.fin.setText(str(self.iterator+1))
        self.snake.setEnabled(False)


    #############################################################
    #################feature matching############################

    def load_image_matching(self,i):
        filename = QtWidgets.QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '':

            self.harris.setText("...")
            
            self.sift.setText("...")
            
            self.ssd.setText("...")
        
            #image=(((Image.open(path)).convert('L')).resize((300,300)))
            image=cv2.imread(path,0)
            image=cv2.resize(image,(255,255),interpolation = cv2.INTER_AREA)
            self.image_mat[i]=image
            
            self.labels[i].setPixmap(QPixmap(path))
            
            

    

    
            
    def matching(self):

        width = (self.image_mat[0]).shape[1]
        height = (self.image_mat[0]).shape[0]


        sift_features = np.zeros((height , len(self.image_mat)*width ) ,np.uint8)
        sift_matches = np.zeros((height , len(self.image_mat)*width ) ,np.uint8)
        
        for i in range(len(self.image_mat)):
 
                keypoints, descriptors,sift_time = sift.computeKeypointsAndDescriptors(self.image_mat[i])
                

                self.keypoints_set[i]=keypoints
                self.descriptors_set[i]=descriptors 
        
        
                sift_features[0:height , i*width : (i + 1)*width ] = self.image_mat[i]

                for j in range ( len ( keypoints ) ) :
                        x , y = keypoints[j].pt
                        cv2.circle ( sift_features , ( (i*width) + int (x) , int (y) ) , 0 , (255 , 0 , 0) , 2)
    
        cv2.imwrite('sift_features.png', sift_features)
        ssd_start=time.time()
        sift_matches = sift_features
        matches, score = siftmatcher(self.keypoints_set,self.descriptors_set)
        #print(score)

        for i in range (int(len(matches)/2))  :
                match = matches.item(i)
                if match != -1:
         
                    cv2.line (sift_matches , (int(((self.keypoints_set[0][i]).pt)[0]) , int(((self.keypoints_set[0][i]).pt)[1])), 
                        (int(((self.keypoints_set[1][match]).pt)[0])+width , int(((self.keypoints_set[1][match]).pt)[1])), 
                        (0,0 , 255 ) , 1 , 0)
    
        cv2.imwrite('sift_matches.png',sift_matches )
    
        self.mat_result.setPixmap(QPixmap('sift_matches.png'))
        ssd_end=time.time()
        ssd_time=(ssd_end - ssd_start)
        self.execution_times(sift_time,ssd_time)

    def execution_times(self,sift_time,ssd_time):
        img1_harris_time=harris(self.image_mat[0])
        img2_harris_time=harris(self.image_mat[1])
        total_time=(img1_harris_time+img2_harris_time)
        self.harris.setText(str(format(total_time,'2f'))+' s')
        self.sift.setText(str(format(sift_time,'2f'))+' s')
        self.ssd.setText(str(format(ssd_time,'2f'))+' s')


    ############################## segmentation #########################

    def load_seg_img(self):
        filename = QtWidgets.QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '':
            self.seg_img2.clear()
            self.seg_label.setText("...")
            self.enable_disable_radio(False)
            image=cv2.imread(path,1)
            self.segmentation_img = image
            self.seg_img1.setPixmap(QPixmap(path))
            self.unchecked_radio(0)
            self.unchecked_radio(1)
            self.H_based.setEnabled(True)
            self.C_based.setEnabled(True)
            self.kmean.setEnabled(False)
            self.region_g.setEnabled(False)
            self.agglomerative.setEnabled(False)
            self.mean_sh.setEnabled(False)
            self.optimal.setEnabled(False)
            self.otsu.setEnabled(False)
            self.spectral.setEnabled(False)


    def optimal_otsu_th(self,img,i):
        image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        id=self.arr_group_radio[1].checkedId()
        
        if id == 0:
            
            if i == 0 :
                self.seg_label.setText(" Global Optimal Thresholding")
                t = optimal_thresholding(image)
                out_image = (image >= t)*image.max()
            elif i == 1:
                self.seg_label.setText("Global Otsu Thresholding")
                t = Otsu_threshold(image)
                out_image = (image >= t)*image.max()
           
            else:
                self.seg_label.setText("Global Spectral Thresholding")
                t_low , t_high =spectral_th(image)
                rows=image.shape[0]
                cols=image.shape[1]
                max_pixel=image.max()
                
                out_image=image
                for i in range(rows):
                    for k in range(cols):
                        pixel = image[i][k]
                        if pixel <= t_low:
                            out_image[i][k]=0
                        elif pixel > (t_high):
                            out_image[i][k]=max_pixel
                        else:
                            out_image[i][k]=max_pixel/2
                            

        if id == 1:
            if i == 0 :
                self.seg_label.setText(" local Optimal Thresholding")
                out_image = local(image, "optimal")
                 
            elif i == 1:
                self.seg_label.setText("local Otsu Thresholding")
                out_image = local(image, "Otsu")
                
            else:
                self.seg_label.setText("local Spectral Thresholding")
                out_image = local(image, "spectral")


        cv2.imwrite('segmented_img.png',out_image )
    
        self.seg_img2.setPixmap(QPixmap('segmented_img.png'))

    
        
    def clustring_seg(self,img,i):
        im=cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
        if i==0:
            self.seg_label.setText("K_Mean Segmentation")
            out = kmean(im)
        elif i==1:
            self.seg_label.setText("Region Growing Segmentation")
            image=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            out = regionGrow(image,6)
        elif i==2:
            self.seg_label.setText("Agglomerative Segmentation")
            out = Agglomerative(im)
        else:
            self.seg_label.setText("Mean Sift Segmentation")
            out = mean_shift(im)
        plt.imshow(out)
        plt.axis('off')
        plt.savefig('segmented_img')
    
        self.seg_img2.setPixmap(QPixmap('segmented_img.png'))

    def radio_checked(self,i):
        if i==0:
            
            self.rad_g.setEnabled(True)
            self.rad_l.setEnabled(True)
            self.optimal.setEnabled(True)
            self.otsu.setEnabled(True)
            self.spectral.setEnabled(True)
            self.radio_G_l[0].setChecked(True)

            self.kmean.setEnabled(False)
            self.region_g.setEnabled(False)
            self.agglomerative.setEnabled(False)
            self.mean_sh.setEnabled(False)
            

        if i == 1:
            
            self.kmean.setEnabled(True)
            self.region_g.setEnabled(True)
            self.agglomerative.setEnabled(True)
            self.mean_sh.setEnabled(True)

            id=self.buttongroup_G_l.checkedId()
            
            self.buttongroup_G_l.setExclusive(False)
            self.radio_G_l[id].setChecked(False)
            self.buttongroup_G_l.setExclusive(True)
            self.optimal.setEnabled(False)
            self.otsu.setEnabled(False)
            self.spectral.setEnabled(False)

        


    def enable_disable_radio(self,i):
        self.H_based.setEnabled(i)
        self.C_based.setEnabled(i)
        
        self.optimal.setEnabled(i)
        self.otsu.setEnabled(i)
        self.spectral.setEnabled(i)
        self.kmean.setEnabled(i)
        self.region_g.setEnabled(i)
        self.agglomerative.setEnabled(i)
        self.mean_sh.setEnabled(i)


    def unchecked_radio(self,group):
        

        id=self.arr_group_radio[group].checkedId()
        self.arr_group_radio[group].setExclusive(False)
        self.arr_arr_radio[group][id].setChecked(False)
        self.arr_group_radio[group].setExclusive(True)
            

    #################### Face Recognition #####################
    def load_rec_img(self):
        filename = QtWidgets.QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '': 
            path_img=pathlib.PurePath(path)
            self.img_path_test=path_img.name 
            print(self.img_path_test)
            self.rec_img_2.clear()
            self.result_btn.clear()
            self.rec_btn.setEnabled(False)
            self.rec_img_1.setPixmap(QPixmap(path))
            self.read_rec_img = cv2.imread(path,1)
            self.dec_btn.setEnabled(True)
            

    def dec_face(self, image):  
        face_detected_img , self.faces = face_detection(image)
        if self.faces != ():
            self.rec_btn.setEnabled(True)
            cv2.imwrite('face_detected.png',face_detected_img )
            self.rec_img_2.setPixmap(QPixmap('face_detected.png'))
        else:
            self.rec_img_2.setText("No faces found")

    def rec_face(self, image ):
            self.result_btn.setText("...")           
            result = recognition(image)
            self.result_btn.setText(str(result))













        

       
    def closeEvent(self , event):

        if(os.path.exists('input_img.png') ):

            os.remove('input_img.png')
        if(os.path.exists('img.png') ):
            os.remove('img.png')
        if(os.path.exists('shapes.png') ):
            os.remove('shapes.png')
        if(os.path.exists('out.jpg') ):
            os.remove('out.jpg')

        if(os.path.exists('sift_features.png') ):
            os.remove('sift_features.png')

        if(os.path.exists('sift_matches.png') ):
            os.remove('sift_matches.png')

        if(os.path.exists('segmented_img.png') ):
            os.remove('segmented_img.png')

        if(os.path.exists('face_detected.png') ):
            os.remove('face_detected.png')


        
    


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
