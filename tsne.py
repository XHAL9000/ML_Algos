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
    mat_prob = np.maximum(mat_prob,1e-12)
    entropy = - mat_prob * np.log(mat_prob)
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
    a = 1e-12 * np.ones(n)  #Lower bound
    b = (2*n)* np.ones(n)  #Upper bound
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
    return np.sum( P * np.log(P/Q) )

def t_qij(q_dist):
    n = q_dist.shape[0]
    coef =  1 / (q_dist + 1)
    for i in range(n) :
        coef[i,i] = 0 # we set qii = 0 since we are only intrested in modeling pairwise similarities
    q_prob = coef / np.sum(coef)
    return np.maximum(q_prob,1e-12)


def gradient(p_prob,y_embed):
    
    q_diff = diff_ij(y_embed)
    q_dist = np.sum(q_diff**2,axis=2)
    q_prob = t_qij(q_dist)
    grad = np.zeros_like(y_embed)
    for i in range(y_embed.shape[0]):
        for j in range(y_embed.shape[0]):
            grad[i,:] += 4*(p_prob[i,j] - q_prob[i,j]) *  (1 / (q_dist[i,j] + 1) )  * q_diff[i,j,:]
    return grad,q_prob
    
def t_sne(X,n_dim=2,prep_wanted=30,lr= 0.5 , epochs = 1000):
    n = X.shape[0]
    p_diff = diff_ij(X)
    p_dist = np.sum(p_diff**2,axis=2) # NxN euclidian distances matrix
    prep,p_prob,var = bisection_var(p_dist,prep_wanted,tol = 0.001, max_iter=500)
    p_prob = (p_prob + p_prob.T) / (2 *n)
    p_prob = 4 * p_prob
    y_embed = 0.0001 * np.random.randn(n,n_dim) # Generate random embeddings with 10E-4 std
    y_embed_old = y_embed
    for itr in range(epochs) : 
        
        grad,q_prob =  gradient(p_prob,y_embed)
                
        if itr < 20:
            momentum = 0.5
        else:
            momentum = 0.8
        
        y_embed_n =  y_embed + lr * grad + momentum * (y_embed - y_embed_old)   
        y_embed_old = y_embed
        y_embed = y_embed_n
        
        if (itr + 1) % 10 == 0:
            C = KL_divergence(p_prob,q_prob)
            print("Iteration ", (itr + 1), ": error is ", C)
            if (itr+1) != 10:
                ratio = C/oldC
                lr = 1/(1+ratio)
                print("ratio ", ratio)
                if ratio >= 0.95:
                    break
            oldC = C
        if itr == 100:
            p_prob = p_prob / 4
        
    
    return y_embed
