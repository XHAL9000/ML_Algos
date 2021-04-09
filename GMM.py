class GMM  : 
    def __init__(self,data,k=3):
        
        self.X = data
        self.k = k
        np.random.rand(self.X.shape[0],self.k)
        self.clusters = np.random.rand(self.X.shape[0],self.k)
        self.pk = np.random.rand(self.k,1)
        self.mu = np.ones((self.k,self.X.shape[1]))
        self.cov = np.ones((self.k,self.X.shape[1],self.X.shape[1]))
        self.initParams()
        
    def initParams(self):
        clusters = np.random.rand(self.X.shape[0],self.k)
        self.clusters = clusters/np.sum(clusters,axis=1,keepdims=1)
        self.pk, self.mu, self.cov  = self.maxim()
    
    def normalProb(self,x,p,mu,cov):
        d = self.X.shape[1]
        inv_cov = np.linalg.pinv(cov)
        det_cov = np.linalg.det(cov) ** 2
        coef1 = 1 /( det_cov * ((2*math.pi)** d/2 ))
        coef = np.dot(np.dot((x-mu).T , inv_cov) , (x-mu) )
        coef2 = math.exp( (-1/2) * coef )
        return coef1 * coef2
    
    def mean(self,w):
        _sum = 0
        for i in range(self.X.shape[0]) :
            _sum = w[i] * self.X[i,:]
        return _sum / np.sum(w)
    
    def covariance(self,mu,w):
        cov = 0
        for i in range(self.X.shape[0]) :
            cov += w[i] * np.dot( (self.X[i,:] - mu).T , self.X[i,:] - mu )
        return cov / np.sum(w)
        
    def expect(self):
        for i in range(self.X.shape[0]):
            for j in range(self.k):
                self.clusters[i,j] = self.normalProb(self.X[i,:],self.pk[j],self.mu[j,:],self.cov[j,:,:])
            self.clusters = self.clusters / np.sum(self.clusters,axis=1,keepdims=1)
        return self.clusters
    
    def maxim(self):
        for j in range(self.k):
            self.pk[j] = sum(self.clusters[:,j]) / self.X.shape[0] 
            self.mu[j,:] = self.mean( self.clusters[:,j] )
            self.cov[j,:,:] = self.covariance(self.mu[j,:] , self.clusters[:,j] )
        return self.pk , self.mu , self.cov
    
    def fit(self,max_iter=10):
        for _ in range(max_iter):
            self.clusters = self.expect()
            self.params = self.maxim()
        print("Training Completed!")
            
