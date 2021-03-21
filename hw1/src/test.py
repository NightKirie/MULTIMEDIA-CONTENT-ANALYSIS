import cv2
import numpy as np
import os
import collections

def count_subarray(arr):
    arr = arr.reshape(-1, arr.shape[-1])
    tuple_arr = [tuple(x) for x in arr]
    return collections.Counter(tuple_arr)
    


a = np.array([[[0,0,0],[0,0,0]],[[0,0,2],[0,0,3]],[[0,0,0],[0,0,0]]])
b = np.array([[[0,0,0],[0,0,1]],[[0,0,1],[0,0,1]],[[0,0,4],[0,0,5]]])


# count_subarray(a)
a_p = count_subarray(a)
b_p = count_subarray(b)
a_p = b_p
print(a_p)
print(b_p)