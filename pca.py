# import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# load the dataset to a datafile
dataset = pd.read_csv('wine.csv')

# distribute the dataset into two components x and y
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# split the x and y into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# perform preprocessing
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# apply PCA function on training and testing set of x component
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_train)
explained_variance = pca.explained_variance_ratio_

# fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predict the test set result using predict function under LogisticRegression
y_pred = classifier.predict(x_test)

# mak confusion matrix between test set y and predicted value
cm = confusion_matrix(y_test, y_pred)

# predict the training set results through scatter plot
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min() - 1, stop=x_set[:,0].max() + 1, step=0.01),
                     np.arange(start=x_set[:,1].min() - 1, stop=x_set[:,1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('yellow', 'white', 'aquamarine')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], color=ListedColormap(('red', 'green', 'blue'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# show scatte plot
plt.show()
