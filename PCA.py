class PCA : 
    def __init__(self,data):
        self.data = data
        self.dataStd = data - np.mean(data, axis = 0, keepdims = True)
        self.cov_mat = np.dot(self.dataStd.T,self.dataStd)
        self.eigVals , self.eigVecs = np.linalg.eig(self.cov_mat)
        
    def percent(self,p=0.9):
        eigValSort =   np.argsort(self.eigVals)
        eigValSum = (self.eigVals**2) / sum(self.eigVals**2)
        dim = 0
        expl = 0
        for i in eigValSort:
            expl += eigValSum[i]
            dim +=1
            if expl >= p:
                return dim
        return data.shape[1]
    
    def fit(self,dim=None,p=0.9):
        if dim == None :
            dim = self.percent(p)
        eigInd = np.argsort(self.eigVals)[:dim]
        eigVecProj = self.eigVecs[:,eigInd]
        dataRed = np.dot(self.dataStd ,eigVecProj)
        return dataRed , self.eigVals[eigInd],eigVecProj
    
