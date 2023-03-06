import numpy as np

n = 10 

X = np.array(range(n))[:, np.newaxis]
X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
print(X_train)