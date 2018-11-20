import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
olivetti = fetch_olivetti_faces()
y = pd.Series(olivetti.target).astype('int').astype('category')
X = pd.DataFrame(olivetti.data)

num_images = X.shape[1]
X.columns = ['pixel_' + str(x) for x in range(num_images)]
Z = []
for i in range(len(y)):
    if y[i] == 7 or y[i] == 9 or y[i] == 31 or y[i] == 34:
        Z.append('F')
    else:
        Z.append('M')

X_values = pd.Series(X.values.ravel())
print(" min: {}, \n max: {}, \n mean: {}, \n median: {}, \n most common \
      value: {}".format(X_values.min(), X_values.max(), X_values.mean(),
      X_values.median(), X_values.value_counts().idxmax()))

for i in range(40):
    first_image = X.loc[(10 * i), :]
    first_label = Z[(10 * i)]
    plottable_image = np.reshape(first_image.values, (64, 64))
    plt.imshow(plottable_image, cmap='gray')
    plt.title('Sexo: {}'.format(first_label))
    plt.show()
