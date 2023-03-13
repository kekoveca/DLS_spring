import numpy as np

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    for batch_start in range(0, len(X) - len(X) % batch_size, batch_size):
        print(perm[batch_start:batch_start+batch_size])
        # print(X[perm[batch_start:batch_start+batch_size]])
        yield X[perm[batch_start:batch_start+batch_size]], y[perm[batch_start:batch_start+batch_size]]

X_fake = np.arange(10)[:, np.newaxis] + 10
y_fake = np.arange(10) + 100

X_reconstructed, y_reconstructed = [], []
for X_batch, y_batch in generate_batches(X_fake, y_fake, 5):
    X_reconstructed.append(X_batch)
    y_reconstructed.append(y_batch)

print(np.array(X_reconstructed))