import numpy as np

def get_wij(X,centers,m):
    n = X.shape[0] #number of data points 
    k = centers.shape[0]  #number of clusters
    W = np.zeros((n,k))   #membership matrix, wij represents the degree that xi belongs to cluster j
    for i in range(n):
        _sum = 0
        for j in range(k):
            W[i,j] = np.linalg.norm(X[i,:] - centers[j,:]) ** (2/(m-1))
            if W[i,j] != 0:
                _sum += 1/W[i,j]
        for l in range(k):
            W[i,l] = W[i,l] *  _sum 
    return 1/W
  
  
  
  def update_clusters (X ,centers,W,m=2) : 
    g = centers.shape[0] #number of clusters
    n = X.shape[0] #number of data points 
    for k in range(g):
        num = 0
        for i in range(n):
            num += W[i,k]*X[i,:]   
        centers[k,:] = num/sum(W[:,k]) #the cluster centroid is a weighted mean of data points
    return centers
  
  
  
  def fit(X,g , m = 2 ,maxIter = 100):
     # g : number of clusters
     # m : model parameter, A large m results in smaller membership values wij, and hence, fuzzier clusters.
    n = X.shape[0] #number of data points
    d = X.shape[1] #number of features
    centers = np.zeros((g,d)) # init cluster means -> centeroids 
    for i in range(g):
        centers[i,:] = X[ np.random.randint(n) ] # select random points as centeroids
    itr = 0
    while itr <  maxIter :
        print(itr)
        W = get_wij(X,centers,m)  # get matrix membership
        centers = update_clusters (X ,centers,W,m) # update clusters
        itr += 1 
    return W , centers
    
