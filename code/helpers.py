# ranking metrics by Mathieu Blondel, November 2013
# License: BSD 3 clause
import numpy as np

def dcg_at_k(r, k, method):
    r = np.asfarray(r)[:k]
    r = 2**r - 1
    if r.size:
        if method == 0:
            #print(r / np.log2(np.arange(2, r.size+2) ))
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size+1) ) )
        #elif method == 1:
        #    return np.sum((r) / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_score(labels, distances, k=10, ks=[2,4,6,8,10]):
    '''
    :param labels: ratings - higher is better
    :param distances: hamming distances, lower is better
    :param k:
    :return: ndcg@k
    '''
    labels = np.array(labels)
    distances = np.array(distances)
    vals = labels[np.argsort(distances)]

    return [_ndcg_at_k(vals, kk) for kk in ks]

def _ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method) # revser=true, because higher ratings should be in the beginning
    if not dcg_max:
        return 0.
    #print(dcg_at_k(r, k, method) , dcg_max)
    return dcg_at_k(r, k, method) / dcg_max

def mrr(labels, distances):
    labels = np.array(labels)
    distances = np.array(distances)

    sorted_dists = np.argsort(distances)

    labels_ordered = labels[sorted_dists].astype(int)

    max_rating = np.max(labels_ordered)
    rank = np.where(labels_ordered == max_rating)[0][0] + 1

    return 1/rank
