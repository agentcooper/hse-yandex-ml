import numpy as np

from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

from sklearn.cluster import KMeans

def printAndWriteAnswer(index, answer):
    print('Answer to ' + str(index) + ': "' + str(answer) + '"')
    f = open('week6-1-' + str(index) + '.txt', 'w')
    f.write(str(answer))
    f.close()

def getClusteredImages(original_image, cluster_labels):
    X_median = np.zeros_like(original_image);
    X_average = np.zeros_like(original_image);

    for k in range(0, n_clusters):
        kth_cluster_indeces = np.array([i for i, v in enumerate(cluster_labels) if v == k])

        kth_cluster_median = np.median(original_image[kth_cluster_indeces], axis=0)
        kth_cluster_average = np.average(original_image[kth_cluster_indeces], axis=0)

        for i in kth_cluster_indeces:
            X_median[i] = kth_cluster_median
            X_average[i] = kth_cluster_average

    return (X_median, X_average)

def PSNR(I, K, size):
    MAX = 1 # because we are using img_as_float

    mse = np.sum(
        [(r*r + g*g + b*b) for (r, g, b) in (I - K)]
    ) / (size * 3)

    return 20 * np.log10(MAX) - 10 * np.log10(mse)

image = imread('parrots.jpg')

n_clusters = 11
clf = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)

I = img_as_float(image)
I_shape = np.shape(I)
I_size = I_shape[0] * I_shape[1]

X = np.reshape(I, (-1, 3))
clf.fit(X)

X_median, X_average = getClusteredImages(X, clf.labels_)

print(
    'Cluster count: {}, PSNR for median: {}, average: {}'.format(
        n_clusters,
        PSNR(X, X_median, I_size),
        PSNR(X, X_average, I_size)
    )
)

printAndWriteAnswer(1, n_clusters)

imsave('parrots_median.jpg', np.reshape(X_median, I_shape))
