import numpy as np

from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

from sklearn.cluster import KMeans

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week6-1-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

def getClusteredImages(X, cluster_labels):
    X_median = np.zeros_like(X);
    X_average = np.zeros_like(X);

    for k in range(0, n_clusters):
        kth_cluster = [i for i, v in enumerate(cluster_labels) if v == k]

        kth_cluster_median = np.median([X[i] for i in kth_cluster], axis=0)
        kth_cluster_average = np.average([X[i] for i in kth_cluster], axis=0)

        for i in kth_cluster:
            X_median[i] = kth_cluster_median
            X_average[i] = kth_cluster_average

    return (X_median, X_average)

def PSNR(I, K):
    MAX = 1 # because we are using img_as_float

    mse = np.sum(
        [(r*r + g*g + b*b) for (r, g, b) in (I - K)]) / (M_width * M_height * 3
    )

    return 20 * np.log10(MAX) - 10 * np.log10(mse)

image = imread('parrots.jpg')

n_clusters = 11
clf = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)

M = img_as_float(image)
M_width, M_height, _ = np.shape(M)

X = np.reshape(M, (-1, 3))
clf.fit(X)

X_median, X_average = getClusteredImages(X, clf.labels_)

print(
    'Cluster count: {}, PSNR for median: {}, average: {}'.format(
        n_clusters,
        PSNR(X, X_median),
        PSNR(X, X_average)
    )
)

printAndWriteAnswer(1, n_clusters)

imsave('parrots_median.jpg', np.reshape(X_median, (M_width, M_height, 3)))
