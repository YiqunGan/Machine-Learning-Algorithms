import numpy as np
x= np.array([[-3,1,1,1,2],[2,2,2,2,2]])
b= np.array([3,2,1,1,1])
preds = np.zeros(2)
z=x+b.T
for i in range(2):
    preds[i - 1] = np.argmax(z[i - 1, :])


print(preds)





