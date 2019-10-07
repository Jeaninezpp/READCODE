import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def basic_clustering(x, n_clusters=10, method='kmeans'):
    """ Initialize a clustering result, i.e., labels and cluster centers.
    :param x: input data, shape=[n_samples, n_features]
    :param method: clustering method, choices=['kmeans','sc','spectral_clustering', 'ac', 'agglomerative_clustering',
                                               'gmm', 'gaussian_mixture_model', 'dbscan', 'ms', 'mean_shift']
    :return: labels and centers
    """

    def get_centers(x, y):
        y_unique = np.sort(np.unique(y[y >= 0]))
        centers = np.zeros([len(y_unique), x.shape[-1]])
        for i, yi in enumerate(y_unique):
            centers[i] = np.mean(x[y == yi], axis=0)
        return centers

    print("Initializing by ", method)
    if method in ['gmm', 'gaussian_mixture_model']:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_clusters, n_init=10)
        gmm.fit(x)
        y_pred = gmm.predict(x)
        centers = gmm.means_.astype(np.float32)
    elif method in ['sc', 'spectral_clustering']:
        from sklearn.cluster import SpectralClustering
        assign_labels = 'discretize'
        affinity = 'nearest_neighbors'
        n_neighbors = 10
        sc = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels,
                                affinity=affinity, n_neighbors=n_neighbors)
        y_pred = sc.fit_predict(X=x)
        centers = get_centers(x, y_pred)
    elif method in ['ac', 'agglomerative_clustering']:
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = ac.fit_predict(X=x)
        centers = get_centers(x, y_pred)
    else:
        if method not in ['km', 'kmeans']:
            print("Using k-means for initialization by default.")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(X=x)
        centers = kmeans.cluster_centers_.astype(np.float32)

    return y_pred, centers
