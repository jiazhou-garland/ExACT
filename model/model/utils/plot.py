import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

import torch
import torchvision.models as models
from torch.nn import functional as F
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import pandas as pd
import seaborn as sns

def plot_tsne(features, labels, file_root):
    '''
    features:(N*m)
    label:(N)
    '''
    x_min, x_max = np.min(features, 1, keepdims=True), np.max(features, 1, keepdims=True)
    features = (features - x_min) / (x_max - x_min)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    class_num = len(labels)
    tsne_features = tsne.fit_transform(features)
    # print('The shape of tsne_features:',tsne_features.shape)
    plt.subplots(figsize=(12, 12))
    df = pd.DataFrame()
    df["y"] = labels[: class_num]
    df["comp-1"] = tsne_features[: class_num,0]
    df["comp-2"] = tsne_features[: class_num,1]
    ax = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    s=15,
                    palette=sns.color_palette("colorblind", len(np.unique(labels))),
                    data=df,
                    marker='.',
                    linewidth=0,
                    legend=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis([-1, 1, -1, 1])
    plt.show()
    plt.savefig(file_root)


def plot_multivariate_normal():
    # define parameters for x and y distributions
    mu_x = 0  # mean of x
    variance_x = 3  # variance of x

    mu_y = 0  # mean of y
    variance_y = 15  # variance of y

    # define a grid for x and y values
    x = np.linspace(-10, 10, 500)  # generate 500 points between -10 and 10 for x
    y = np.linspace(-10, 10, 500)  # generate 500 points between -10 and 10 for y
    X, Y = np.meshgrid(x, y)  # create a grid for (x,y) pairs

    # create an empty array of the same shape as X to hold the (x, y) coordinates
    pos = np.empty(X.shape + (2,))

    # fill the pos array with the x and y coordinates
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # create a multivariate normal distribution using the defined parameters
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

    # create a new figure for 3D plot
    fig = plt.figure()

    # add a 3D subplot to the figure
    ax = fig.add_subplot(projection='3d')

    # create a 3D surface plot of the multivariate normal distribution
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)

    # set labels for the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # display the 3D plot
    plt.show()
    return