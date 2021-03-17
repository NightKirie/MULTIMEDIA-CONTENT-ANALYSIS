from skimage.measure import block_reduce
import numpy as np

a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
print(block_reduce(a, block_size=(2,1), func=np.mean))