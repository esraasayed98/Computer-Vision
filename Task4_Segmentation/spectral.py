from otsu import Otsu_threshold
import numpy as np

def spectral_th(image):
    image=np.array(image)
                
    image_f=image.flatten()
    t=Otsu_threshold(image_f)
    image_lower=[]
    image_higher=[]
    for i in range(len(image_f)):
        if image_f[i]<t:
            image_lower.append(image_f[i])
        else:
            image_higher.append(image_f[i])

    t_low=Otsu_threshold(np.array(image_lower))
    t_high=Otsu_threshold(np.array(image_higher))

    

    return t_low, t_high
