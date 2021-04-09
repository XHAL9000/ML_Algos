############ UTILS ###############""
import math
##Factorial##
def factorial(x):
    if x<0 : 
        print("x is less than 0") ### raise ERROR
    if x == 0 :
        return 1
    return x*factorial(x-1)
####################

######## DISCRETE FUNCTION ############

class Bernoulli:
    def __init__(self, prob = 0.5):
        if 0.0 <= prob and prob <= 1.0 :
            self.p = prob
            self.mean = self.mean()
            self.variance = self.variance()
        else :  
            print("invalid bernoulli value")

    def pmf(self, x):
        if x == 0:
            return 0
        elif x == 1:
            return 1
        else:
            raise Exception("x value is not contained within bernoulli range of {0,1}")
    def mean(self):
        return self.p
    def variance(self):
        return self.p * (1.0 - self.p)
    

    
class Binomial:
    def __init__(self,n=30,prob = 0.5):
        if 0.0 <= prob and prob <= 1.0 :
            self.p = prob
            self.n = n
            self.mean = self.mean()
            self.variance = self.var()
        else :  
            raise Exception("invalid bernoulli value")

    def pmf(self, x):
        if int(x)== x and x >=0 and x <= self.n :
            k = self.n-x
            coef = factorial(self.n)/(factorial(x) * factorial(k))
            return coef * (self.p**x)*((1-self.p)**k)
        else:
            raise Exception("x value is not contained within binomial range of {0,n={}} or is not an integer".format(self.n))
    def mean(self):
        return self.n*self.p
    def variance(self):
        return self.n*self.p * (1.0 - self.p)
    
class Poisson:
    def __init__(self,theta = 0.5):
        if 0.0 < theta :
            self.theta = theta
            self.mean = self.esperance()
            self.variance = self.var()
        else :  
            raise Exception("invalid Poisson theta value")

    def pmf(self, x):
        if int(x)==x :
            pr = math.exp(-self.theta) * ((self.theta)**x) / factorial(x)
            return pr
        else:
            raise Exception("x = {} is not an integer ".format(x))
    def esperance(self):
        return self.theta
    def var(self):
        return self.theta
    
    
class Geometry:
    def __init__(self,p = 0.5):
        if 0.0 < p and p < 1 :
            self.p = p
            self.mean = self.esperance()
            self.variance = self.var()
        else :  
            raise Exception("invalid Geometry p value")

    def pmf(self, x):
        if int(x)==x and x>0:
            pr = self.p * ((1-self.p) ** (x-1)) 
            return pr
        else:
            raise Exception("x = {} is not an integer or x is 0 ".format(x))
    def esperance(self):
        return 1/self.p
    def var(self):
        return (1-self.p)/self.p**2
    
class Uniform:
    def __init__(self,n):
        if 0.0 < n and int(n)==n   :
            self.n = n
            self.mean = self.esperance()
            self.variance = self.var()
        else :  
            raise Exception("invalid n value")

    def pmf(self, x):
        if int(x)==x and x>0 and x <=self.n:
            pr = 1/self.n
            return pr
        else:
            raise Exception("x = {} is not an integer or x is 0 ".format(x))
    def esperance(self):
        return (self.n+1)/2
    def var(self):
        return (self.n**2 - 1)/12
    
class HyperGeometry:
    def __init__(self,N,n,p):
        if 0.0 < n and int(n)==n and int(N)==N and 0.0<N and n<=N and 0.0 < p and p < 1  :
            self.n = n
            self.N = N
            self.p = p
            self.mean = self.esperance()
            self.variance = self.var()
        else :  
            raise Exception("invalid N or n or p value")

    def pmf(self, x):
        if int(x)==x and x<= min(self.N*self.p,self.n) and x >= max(0,self.n-self.N+self.N*self.p) :
            coef1 = factorial(self.N*self.p) / (factorial(self.N*self.p-x) * factorial(x))
            coef2 = factorial(self.N-self.N*self.p) / (factorial(self.N-self.N*self.p-(self.n-x)) * factorial(self.n-x))
            coef3 = factorial(self.N) / (factorial(self.N-self.n) * factorial(self.n))
            return coef1*coef2/coef3
        else:
            raise Exception("x = {} is not an integer or x is not valid ".format(x))
    def esperance(self):
        return self.n*self.p
    def var(self):
        return self.n*self.p*(self.N-self.N*self.p)*(self.N-self.n)/(self.N*(self.N-1))
    

    
#### Continiuos Probablity ########

class Normal:
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        self.mean = self.esperance()
        self.variance = self.var()


    def pmf(self, x):
        coef1 = 1/(self.sigma * math.sqrt(2*math.pi))
        coef2 = (x-self.mu / self.sigma) **2
        return coef1* math.exp(-(1/2) *coef2)

    def cdf(self, x):
        z = (x-self.mu)/(self.sigma*math.sqrt(2))
        return (1/2)* (1+math.erf(z))
    
    def esperance(self):
        return self.mu
    
    def var(self):
        return self.sigma**2
    
    
class Exponential:
    def __init__(self,theta):
        if theta > 0 :
            self.theta = theta
            self.mean = self.esperance()
            self.variance = self.var()
        else:
            print("Not a valid Theta = {}".format(theta) )


    def pmf(self, x):
        if x<0:
            return 0
        else:
            return self.theta * math.exp(-self.theta*x)

    def cdf(self, x):
        if x<0:
            return 0
        else:
            return 1- math.exp(-self.theta*x)

    def esperance(self):
        return 1/self.theta
    
    def var(self):
        return 1/(self.theta**2)
    
class UniformC:
    def __init__(self,a,b):
        if a <= b :
            self.a = a
            self.b = b
        else:
            self.a = b
            self.b = a
        self.mean = self.esperance()
        self.variance = self.var()

    def pmf(self, x):
        if x<=self.b and x>= self.a:
            return 1/(self.b-self.a)
        else:
            return 0

    def cdf(self, x):
        if x<=self.b and x>= self.a:
            return (x-self.a)/(b-self.a)
        elif x<=a:
            return 0
        else:
            return 1
        
    def esperance(self):
        return (self.a + self.b)/2
    
    def var(self):
        return (self.b - self.a)**2 / 12

    
class ChiSquare:
    def __init__(self,k):
        if k==int(k) and k>0 :
            self.k = k
        else:
            raise Exception("k = {} must be an integer > 0".format(k))
        self.mean = self.esperance()
        self.variance = self.var()

    def pmf(self, x):
        if x>0:
            k=self.k
            coef1 = 2**(k/2)* (math.gamma(k/2))
            coef2 = x**((k/2) - 1) * math.exp(-x/2)
            return coef2 /coef1 
        else:
            return 0
            
    def cdf(self, x):
        if x>0:
            k=self.k
            coef1 = special.gammainc(k/2,x/2)
            coef2 =  math.gamma(k/2)
            return coef1 / coef2 
        else:
            return 0
        
    def esperance(self):
        return self.k
    
    def var(self):
        return 2*self.k
