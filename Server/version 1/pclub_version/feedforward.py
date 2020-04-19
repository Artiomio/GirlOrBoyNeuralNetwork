import numpy as np
@np.vectorize
def relu(x):
    return max(x, 0)


def feedforward(inputs, layers, w):
	a = inputs # Сначала inputs
	for (i, layer) in enumerate(layers):
		a = np.append(a, 1)
		a = relu(np.dot(w[i], a))
	return a
	