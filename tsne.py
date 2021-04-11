
def diff_ij(Y):
    """
    Tensor of  vector differences 
    :param Y:data embedding Matrix  NxK 
    :return: Tensor of differences NxNxK matrix
    """
    n = Y.shape[0]
    k = Y.shape[1]
    diff = np.zeros((n,n,k))
    n = X.shape[0]
    for i in range(n):
        for j in range(i,n):
            diff[i,j,:] = Y[i,:]-Y[j,:]
            diff[j,i,:] =  - diff[i,j,:]
    return diff


def prob_ij(mat_dist,var):
    n = mat_dist.shape[0]
    mat_prob = np.zeros((n,n))
    for i in range(n) :
        coef = -  mat_dist[i,:] / (2 * var[i])
        coef = np.exp(coef)
        coef[i] = 0 # we set pii = 0 since we are only intrested in modeling pairwise similarities
        mat_prob[i,:] = coef / (np.sum(coef))
    return mat_prob

def _preplixity(mat_dist,var):
    """
    Compute the preplexity given variances
    """
    mat_prob = prob_ij(mat_dist,var)
    entropy = - mat_prob * np.log(mat_prob+0.001)  # add 0.001 because mat_prob[i,i] = 0
    entropy = np.sum(entropy,axis=1)
    return 2**entropy,mat_prob

    
def bisection_var(mat_dist,prep_wanted=30.0,tol = 0.01, max_iter=30):
    """
    Using the bisection algorithm to find variances wrt to a given preplixity
    """
    n = mat_dist.shape[0]
    var = np.std(mat_dist,axis=1)
    prep,mat_prob = _preplixity(mat_dist,var)
    diff_prep = prep - prep_wanted
    niter = 0
    a = 0.001 * np.ones(n)  #Lower bound
    b = (n*2)* np.ones(n)  #Upper bound
    while np.linalg.norm( diff_prep ) > tol and  niter < max_iter:
        for i in range(n) : 
            if  diff_prep[i] <  0 :
                a[i] = var[i]  # Replace the lower bound
            elif diff_prep[i] >  0 :
                b[i] = var[i] # Replace the Upper bound
            var[i] = (a[i]+b[i]) / 2
        prep,mat_prob = _preplixity(mat_dist,var)
        diff_prep = prep - prep_wanted
        niter += 1
    print(niter)
    return prep,mat_prob,var


def KL_divergence(P,Q):
    return np.sum( P * np.log(P/Q +0.001 ) )

def t_qij(q_dist):
    n = q_dist.shape[0]
    coef =  1 / (q_dist + 1)
    coef = np.exp(coef)
    for i in range(n) :
        coef[i,i] = 0 # we set qii = 0 since we are only intrested in modeling pairwise similarities
    q_prob = coef / np.sum(coef)
    return q_prob


def gradient(p_prob,y_embed):
    q_diff = diff_ij(y_embed)
    q_dist = np.sum(q_diff**2,axis=2)
    q_prob = t_qij(q_dist)
    grad = np.dot(np.dot((p_prob - q_prob) , q_diff).T ,   (1 / (q_dist + 1) )) 
    grad =  4 * np.sum(grad, axis=1)
    return grad.T
    
def t_sne(X,n_dim=2,prep_wanted=10,lr= 0.1 , momentum = 0.5, epochs = 1000):
    n = X.shape[0]
    p_diff = diff_ij(X)
    p_dist = np.sum(p_diff**2,axis=2) # NxN euclidian distances matrix
    prep,p_prob,var = bisection_var(p_dist,prep_wanted,tol = 0.001, max_iter=500)
    p_prob = (p_prob + p_prob.T) / (2 *n)
    y_embed = 0.0001 * np.random.randn(n,n_dim) # Generate random embeddings with 10E-4 std
    y_embed_old = y_embed
    for i in range(epochs) : 
        grad=  gradient(p_prob,y_embed)
        y_embed_n =  y_embed + lr * grad + momentum * (y_embed - y_embed_old)   
        y_embed_old = y_embed
        y_embed = y_embed_n
    
    return y_embed
