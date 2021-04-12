def sigmoid(x):
    x = np.dot(x, x.T)
    return 1/(1+np.exp(-x))

def linear(x):
    x = np.dot(x, x.T)
    return x

def rbf(x, gamma = 10):
    n,d = x.shape
    mat_dists = np.zeros((n,n))
    for i in range(n):
        mat_dists[i,i] = 0
        for j in range(i+1,n):
            dist = x[i,:]-x[j,:] 
            mat_dists[i,j] = np.dot( dist.T , dist )
            mat_dists[j,i] = mat_dists[i,j]            
    return np.exp(-gamma*mat_dists)

def kpca(data, n_dims=2, kernel = rbf):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target n_dims
    :param kernel: kernel functions
    :return: (n_samples, n_dims)
    '''

    K = kernel(data)
    #
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    Kern = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    #
    eig_values, eig_vector = np.linalg.eig(K)
    idx = eig_values.argsort()[::-1]
    eigval = eig_values[idx][:n_dims]
    eigvector = eig_vector[:, idx][:, :n_dims]
    print(eigval)
    eigval = eigval**(1/2)
    vi = eigvector/eigval.reshape(-1,n_dims)
    data_n = np.dot(K, vi)
    return data_n
