import numpy as np
import math
import queue


NOISE = 0
UNASSIGNED = -1

def dist(a, b):
    """
    Euclidian Distance
    :param a: vector
    :param b: vector
    :return: euclidian distance
    """
    return np.linalg.norm(a-b)

def neighbors_mat(X,eps):
    """
    Compute distances between instances and then get neighbors wrt epsilon
    :param X: Data matrix  NxD 
    :param eps: radius
    :return: a square matrix NxN = Adjacency matrix  1 if two instances are close else 0
    """
    mat_nbrs = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]) : 
        for j in range(i+1,X.shape[0]):
            if dist(X[i,:],X[j,:]) <= eps :
                mat_nbrs[j,i] = 1
                mat_nbrs[i,j] = 1
    return mat_nbrs
    

def get_core_id(mat_nbrs,minPts):
    """
    get core indices of X wrt epsilon and minPts
    :param mat_nbrs: Neighbors matrix = Adjacency matrix NxN
    :param minPts: Minimum Points to define a core instance
    :return: Indices of core instances 
    """
    coresId = np.zeros(mat_nbrs.shape[0])
    for i in range(mat_nbrs.shape[0]) : 
        if sum(mat_nbrs[i,:]) >= minPts :
            coresId[i] = 1
    return coresId


def assign_cls(mat_nbrs, clusters, cores ,pntId, clsId, eps , minPts):
    """
    
    :param mat_nbrs : Neighbors matrix = Adjacency matrix NxN
    :param clusters: List of cluster assignement. 1xN
    :param pntId: instance Id
    :param clsId : cluster Id
    :param eps: radius
    :param minPts: Minimum Points to define a core instance
    :return: False if instance of pntId is not core else True. 
    """
    
    q = queue.Queue()
    q.put(pntId)

    while not q.empty():
        qId = q.get()
        if cores[qId] == 1 :        
            neibrs = np.where( mat_nbrs[qId,:] == 1)[0]
            for i in neibrs:
                resultId = i
                if clusters[resultId] == UNASSIGNED:
                    q.put(resultId)
                    clusters[resultId] = clsId
                elif clusters[resultId] == NOISE:
                    clusters[resultId] = clsId
    return clusters


def dbscan(X, eps, minPts):
    """
    DBSCAN algorithm
    :param X: data matrix NxD
    :param eps: epsilon the radius of the minimum distance
    :param minPts:  minimum points to define a core point
    :return: a list of cluster assignement and numberof clusters
    """
    
    mat_nbrs = neighbors_mat(X,eps)
    cores = get_core_id(mat_nbrs,minPts)
    print(cores)
    clsId = 1
    nPoints = X.shape[0]
    clusters = [UNASSIGNED] * nPoints
    for ptId in range(nPoints):
        if clusters[ptId] == UNASSIGNED:
            if cores[ptId] == 0 :
                clusters[ptId] = NOISE
            else :
                clusters[ptId] = clsId
                clusters = assign_cls(mat_nbrs, clusters,cores, ptId, clsId, eps , minPts)
                clsId = clsId + 1
    return clusters , clsId



