from PIL import Image, ImageDraw
import numpy as np




def padd_with_first_col_row(img,shape):
		
		input_shape=img.shape
		
		new=img
		# l_pixel=arr[:,0]
		# r_pixel=arr[:,shape[1]-1]

		l_pixel=img[:,0]
		r_pixel=img[:,input_shape[1]-1]

		padd_x=int((shape[1]-input_shape[1])/2)
		#print(input_shape)
		for i in range(padd_x):
			#rewrite the last col
			new=np.hstack((new,np.atleast_2d(r_pixel).T))
			new=np.hstack((np.atleast_2d(l_pixel).T,new))
			


		t_pixel=new[0,:]
		b_pixel=new[input_shape[0]-1,:]
		padd_y=int((shape[0]-input_shape[0])/2)
		for i in range(padd_y):
			#rewrite the last col
			new=np.vstack((new,(b_pixel)))
			new=np.vstack(((t_pixel),new))

		return new



# image=Image.open('77.png').convert('L')
# arr=np.asarray(image)
# shape=arr.shape

# i_s=(330,330)
# #new=np.asarray(image)
# i=padd_with_first_col_row(arr,i_s)
# img_save=Image.fromarray(i)
# print(shape)
# print(i.shape)
# img_save.show()