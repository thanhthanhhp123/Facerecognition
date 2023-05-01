from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def visualize_3d(X, y):
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap = 'rainbow')
    ax.set_title('3D Visualization')
    ax.view_init(20, 60)
    plt.show()

def visualize_2d(X, y):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker = 9, cmap = 'rainbow')
    plt.show()

def visualize(X, y, classes):
    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            index = random.randint(0, len(X))
            ax[i, j].imshow(X[index], cmap = 'gray')
            ax[i, j].set_title(classes[y[index]])
            ax[i, j].axis('off')
    plt.show()