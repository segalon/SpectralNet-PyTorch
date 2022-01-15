import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from sklearn.cluster import KMeans

def get_affinities(X, params, to_torch=False):
  if params['affinity'] == 'rbf':
    dists = torch.cdist(X,X)
    W = torch.exp(-1 * params['gamma'] * (dists ** 2))
    W.fill_diagonal_(0)
    return W
      
  # variation of https://github.com/KlugerLab/SpectralNet/blob/e0993413ba4604258f979496e292fdcd26ea7a18/src/core/costs.py
  elif params['affinity'] == "ngb":
    n_ngb = params['n_ngb']
    m = X.shape[0]
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=float)
        
    dists = torch.cdist(X,X)#.toarray()
    nn = torch.topk(-dists, n_ngb, sorted=True)
    vals = nn[0]
    scales = -vals[:, - 1]
    const = X.shape[0] // 2
    scales = torch.topk(scales, const)[0]
    scale = scales[const - 1]
    vals = vals / (2 * scale)
    aff = torch.exp(vals)

    idx = nn[1]
 
    W = torch.zeros(m, m)
    W = W.float()
    aff = aff.float()
    W[np.arange(m)[:, None], idx] = aff

    W = 0.5 * (W + W.T)
    W.fill_diagonal_(0)
    if to_torch:
        return W
    else:
        return W.cpu().detach().numpy()



# from https://github.com/KlugerLab/SpectralNet/blob/e0993413ba4604258f979496e292fdcd26ea7a18/src/core/data.py
def generate_cc(n=1200, noise_sigma=0.1, train_set_fraction=1.):
    '''
    Generates and returns the nested 'C' example dataset (as seen in the leftmost
    graph in Fig. 1)
    '''
    pts_per_cluster = int(n / 2)
    r = 1

    # generate clusters
    theta1 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)
    theta2 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)

    cluster1 = np.concatenate((np.cos(theta1) * r, np.sin(theta1) * r), axis=1)
    cluster2 = np.concatenate((np.cos(theta2) * r, np.sin(theta2) * r), axis=1)

    # shift and reverse cluster 2
    cluster2[:, 0] = -cluster2[:, 0] + 0.5
    cluster2[:, 1] = -cluster2[:, 1] - 1

    # combine clusters
    x = np.concatenate((cluster1, cluster2), axis=0)

    # add noise to x
    x = x + np.random.randn(x.shape[0], 2) * noise_sigma

    # generate labels
    y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)

    # shuffle
    p = np.random.permutation(n)
    y = y[p]
    x = x[p]

    # make train and test splits
    n_train = int(n * train_set_fraction)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train].flatten(), y[n_train:].flatten()

    return x_train, x_test, y_train, y_test



# by https://github.com/ChongYou/subspace-clustering/blob/master/metrics/cluster/accuracy.py
def clustering_accuracy(labels_true, labels_pred):
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
    


def get_cluster_acc(model, data, k):
  N = len(data)
  loader_for_eval = torch.utils.data.DataLoader(data, batch_size=N, shuffle=False)
  
  with torch.no_grad():
    model.eval()
    for batch_idx, (data, target) in enumerate(loader_for_eval):
        data = data.view(N, input_sz)
        #data, target = data.to(device), target.to(device)
        Y_e = model(data)
        labels_true = target
    Y_e = Y_e.cpu().detach().numpy()
    labels_true = labels_true.cpu().detach().numpy()
    km = KMeans(n_clusters=k, random_state=20)
    cluster_preds = km.fit_predict(Y_e)
    # measure the clustering accuracy
    acc = clustering_accuracy(labels_true, cluster_preds)
    return acc



