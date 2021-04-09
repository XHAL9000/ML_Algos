class Data:
    def __init__(self,data):
        if isinstance(data,pd.DataFrame):
            self.data = data.to_numpy()
        if isinstance(data,np.ndarray):
            self.data = data
        self.shape = self.data.shape
        self.row = self.shape[0]
        self.col = self.shape[1]
        
    def mean(self, col, weight = [None]):
        #validate(data, col)
        n = self.row
        if any(weight) != None :
            return sum(self.data[:,col]) / n
        else :
            assert len(weight) == n, print("Not valid weight size")
            _sum = np.dot(self.data[:,col],weight)
            return _sum/sum(weight)
            

    def variance(self,col):
        n= self.row
        _mean = self.mean(col)
        return sum((self.data[:,col] - _mean)**2) / n
    
    def covariance(self):
        n = self.row
        d = self.col
        covar_mat = np.zeros([d,d])
        for i in range(0,d):
            covar_mat[i,i] = self.variance(i)
            for j in range (i+1,d):
                _meani = self.mean(i)
                _meanj = self.mean(j)
                cov_ij = sum((self.data[:,i] - _meani)*(self.data[:,j] - _meanj)) / n
                covar_mat[i,j] = cov_ij
                covar_mat[j,i] = cov_ij
        return covar_mat
    
    def correlation(self):
        d = self.col
        covar_mat = self.covariance()
        corr_mat = np.zeros([d,d])
        for i in range(0,d):
            corr_mat[i,i] = 1
            for j in range (i+1,d):
                 corr_mat[i,j] = covar_mat[i,j] / (math.sqrt(covar_mat[j,j]*covar_mat[i,i]))
                 corr_mat[j,i] = corr_mat[i,j]
        return corr_mat
    
    def standarize(self):
        d = self.col
        StanData = np.zeros(self.shape)
        cov_mat = self.covariance()
        for i in range(0,d):
            StanData[:,i] = (self.data[:,i]-self.mean(i))/math.sqrt(cov_mat[i,i])
        return StanData
    
    def eigen(self):
        return np.linalg.eig(self.data)
