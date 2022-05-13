import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
from copy import deepcopy



cluster = {}
centers = {}

###############---------------   FUNCTIONS   ----------------##################
def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.
    
    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two clusters.
    
    cluster1 and cluster2 are lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)

    

def initial_clusters(points):

    
    groups = {}
    d = int(256 / (20))
    for i in range(20):
        j = i * d
        groups[(j, j, j)] = []
    for i, p in enumerate(points):
        go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
        groups[go].append(p)
    return [g for g in groups.values() if len(g) > 0]
    
def fit(points,k):

    clusters_list = initial_clusters(points)

    while len(clusters_list) > k:

        # Find the closest (most similar) pair of clusters
        cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
                key=lambda c: clusters_distance_2(c[0], c[1]))

        # Remove the two clusters from the clusters list
        clusters_list = [c for c in clusters_list if c != cluster1 and c != cluster2]

        # Merge the two clusters
        merged_cluster = cluster1 + cluster2

        # Add the merged cluster to the clusters list
        clusters_list.append(merged_cluster)


    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            cluster[tuple(point)] = cl_num
            
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)
          


def predict_cluster(point):
    """
    Find cluster number of point
    """
    # assuming point belongs to clusters that were computed by fit functions
    return cluster[tuple(point)]

def predict_center(point):
    """
    Find center of the cluster that point belongs to
    """
    point_cluster_num = predict_cluster(point)
    center = centers[point_cluster_num]
    return center

def Agglomerative(img):
    pixels = img.reshape((-1,3))
    fit(pixels,4)
    new_img = [[predict_center(list(pixel)) for pixel in row] for row in img]
    new_img_luv = np.array(new_img, np.uint8)
    return new_img_luv

#Agglomerative(luv)

