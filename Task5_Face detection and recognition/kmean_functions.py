import numpy as np
import random
import pandas as pd 


def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5


def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid



def calc_centroids(clusters, X):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],
                      axis=1)

    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids




def kmean(im):
    shape=im.shape
    im=(im/255).reshape(shape[0]*shape[1],3)
    random_index = random.sample(range(0, len(im)), 2)

    centroids = []
    for i in random_index:
        centroids.append(im[i])
    centroids = np.array(centroids)

    
    for i in range(5):
        get_centroids = findClosestCentroids(centroids, im)
        centroids = calc_centroids(get_centroids, im)

    im_recovered = im.copy()
    for i in range(len(im)):
        im_recovered[i] = centroids[get_centroids[i]]
        

    im_recovered = im_recovered.reshape(shape[0], shape[1], 3)

    return im_recovered
    