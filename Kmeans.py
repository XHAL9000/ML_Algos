class Kmeans  : 
    def __init__(self,data,k=3,kmeans = None):
        
        self.X = data
        self.k = k
        self.clusters = np.zeros(self.X.shape[0])
        if hasattr(kmeans, '__array__'):
            self.centers = np.asarray(kmeans, dtype=np.float)
        else :
            self.centers = self.initCenters()

        
    def initCenters(self):
        _min = np.min(self.X, axis = 0)
        _max  = np.max(self.X, axis = 0)
        centers = np.empty((self.k,self.X.shape[1]))
        for i in range(self.k):
            centers[i,:] = self.X[ np.random.randint(self.X.shape[0]) ]
            #centers[i,:] = _min + (_max -_min) * np.random.randn()
        return centers
    
    def euclid_dist(self,X,Y):
        return np.linalg.norm(X-Y)
    
    def assign(self):
        for i in range(self.X.shape[0]):
            dist_min = 999999
            for j in range(self.k):
                dist_ij = np.linalg.norm(self.X[i,:]-self.centers[j,:])
                if dist_ij < dist_min :
                    dist_min = dist_ij
                    self.clusters[i] = j
        return self.clusters
    
    def fit(self,max_iter=10):
        for _ in range(max_iter):
            self.clusters = self.assign()
            for i in range(self.k):
                dataClustk = self.X[self.clusters==i,:]
                self.centers[i,:] = np.mean(dataClustk,axis=0)
        return self.centers,self.clusters
            
