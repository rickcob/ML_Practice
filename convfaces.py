import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D
TRAIN_TEST_SPLIT = 0.9

olivetti = fetch_olivetti_faces()
y = pd.Series(olivetti.target).astype('int').astype('category')
X = pd.DataFrame(olivetti.data)
mask = np.random.rand(len(X)) < TRAIN_TEST_SPLIT
X_tr = X[mask]
X_te = X[~mask]
y_tr = y[mask]
y_te = y[~mask]
Z_tr = []
Z_te = []

for i in range(len(y_tr)):
    if y_tr[i] == 7 or y_tr[i] == 9 or y_tr[i] == 31 or y_tr[i] == 34:
        Z_tr.append(-1)
    else:
        Z_tr.append(1)
for i in range(len(y_te)):
    if y_te[i] == 7 or y_te[i] == 9 or y_te[i] == 31 or y_te[i] == 34:
        Z_te.append(-1)
    else:
        Z_te.append(1)

X_tr = np.array(X_tr)
X_tr = X_tr.reshape(X_tr.shape[0], 64, 64, 1)
X_te = np.array(X_te)
X_te = X_te.reshape(X_te.shape[0], 64, 64, 1)
Z_te = np.array(Z_te)
Z_te = Z_te.reshape(Z_te.shape[0], 1)

convnet = Sequential()
convnet.add(Convolution2D(1, (2, 2), strides=(2, 2), activation='tanh',
                          input_shape=(64, 64, 1)))
convnet.add(Convolution2D(4, (6, 6), strides=(2, 2), activation='tanh'))
convnet.add(Convolution2D(16, (6, 6), strides=(2, 2), activation='tanh'))
convnet.add(Convolution2D(64, (5, 5), activation='tanh'))
convnet.add(Flatten())
convnet.add(Dense(1, activation='tanh'))

convnet.compile(loss='mean_squared_error', optimizer='sgd',
                metrics=['accuracy'])

convnet.fit(X_tr, Z_tr, batch_size=32, nb_epoch=10,
            verbose=1)

metrics = convnet.evaluate(X_te, Z_te, verbose=1)
print()
print("%s: %.2f%%" % (convnet.metrics_names[1], metrics[1]*100))
predictions = convnet.predict(X_te)
for i in range(len(predictions)):
    if predictions[i] < 0:
        predictions[i] = -1
    else:
        predictions[i] = 1
predictions = np.append(Z_te, predictions, axis=1)
print(predictions)
