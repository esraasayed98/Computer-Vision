
import numpy as np


SIFT_MATCH_THRESHOLD = 200
 
def siftmatcher(keypoints_set , descriptors_set):
	keypoint_1 = keypoints_set[0]
	keypoint_2 = keypoints_set[1]
	descriptor_1 = descriptors_set[0]
	descriptor_2 = descriptors_set[1]
 
	diff = descriptor_2 - descriptor_1[:,None]
	squre_of_diff = diff ** 2
	sum_square_diff = squre_of_diff.sum(axis=-1)
	score = np.sqrt(sum_square_diff.min(axis=-1))
	
	matches = np.argmin(sum_square_diff,axis=-1)
	invalid_matches = score > SIFT_MATCH_THRESHOLD 
	
	score[invalid_matches] = -1
	matches[invalid_matches] = -1
 
	return matches , score