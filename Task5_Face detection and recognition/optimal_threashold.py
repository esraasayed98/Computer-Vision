import numpy as np



def optimal_thresholding(img):
    
    back_ground_sum = (img[0,0] + img[0,-1] + img[-1,0] + img[-1,-1])
    
    foren_ground_sum = np.sum(img) - back_ground_sum
    
    back_ground_mean = back_ground_sum / 4
    

    foren_ground_mean = foren_ground_sum / (np.size(img)-4)
    
    t = (back_ground_mean + foren_ground_mean) / 2
    
    while True:
        back_ground_mean = np.mean(img[img < t])
        
        foren_ground_mean = np.mean(img[img >= t])
        
        if t == ((back_ground_mean + foren_ground_mean) / 2):
            break
        t = (back_ground_mean + foren_ground_mean) / 2
    return t
    
