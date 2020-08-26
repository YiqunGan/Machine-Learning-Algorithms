import numpy as np
param = dict()

a=np.array([1,2,-1,-1])
b=np.maximum(a,0)
b[b>0] =1
print(a)
