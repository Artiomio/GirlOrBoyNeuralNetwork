import numpy as np

@np.vectorize
def relu(x):
    return max(x, 0)


def feedforward(inputs, w):
	a = inputs # Сначала inputs
	for i in range(0, len(w)):
		a = np.append(a, 1)
		a = relu(np.dot(w[i], a))
	return a
	