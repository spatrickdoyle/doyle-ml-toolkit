#Collection of feature extraction functions
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University

from abc import ABCMeta, abstractmethod
import numpy as np

PI = np.pi
E = np.e

class Extractor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        #Set the string name of the method - that's it

        pass

    @abstractmethod
    def getC(n, X):
        #int n: the order of the coefficient to generate
        #list X[]: the data sweep being approximated
        #returns the nth order coefficient of the given decomposition of X
        #OR it could return a tuple of two associated coefficients

        pass

    @abstractmethod
    def evalH(C, T, x):
        #list C[]: list of coefficients to use in this approximation
        #float T: the period on which the function is approximated, [0,T]
        #float x: value to evaluate the function at
        #returns the result of the given approximation evaluated at x

        pass


#Complex Fourier decomposition, discontinuous
class ComplexFourier(Extractor):
    def __init__(self):
        self.name = "ComplexFourier"

    def getC(self, n, X):
        #Use the data set to figure out omega and tau and such
        m = len(X) #Length of the set
        T = len(X) #Period of the set
        w = (2*PI)/T #Angular frequency

        #Calculate and return the coefficients - bearing in mind the formula for the 0th is different
        if n == 0:
            return (1.0/(2.0*T))*sum([X[j]+X[j-1] for j in range(1,m)])
        else:
            part1 = -(1/(1j*n*w*T))*sum([((X[l]-X[l-1])/((l+1)-l))*((l+1)*(E**(-1j*n*w*(l+1))) - l*(E**(-1j*n*w*l))) + (E**(-1j*n*w*(l+1)) - E**(-1j*n*w*l))*(X[l]-((X[l]-X[l-1])/((l+1)-l))*(l+1)+(1/(1j*n*w))*((X[l]-X[l-1])/((l+1)-l))) for l in range(1,m)])
            n *= -1
            part2 = -(1/(1j*n*w*T))*sum([((X[l]-X[l-1])/((l+1)-l))*((l+1)*(E**(-1j*n*w*(l+1))) - l*(E**(-1j*n*w*l))) + (E**(-1j*n*w*(l+1)) - E**(-1j*n*w*l))*(X[l]-((X[l]-X[l-1])/((l+1)-l))*(l+1)+(1/(1j*n*w))*((X[l]-X[l-1])/((l+1)-l))) for l in range(1,m)])
            return (part1,part2)

    def evalH(self, C, T, x):
        #Use T to calculate omega
        w = (2*PI)/T
        order = len(C)-1

        #Evaluate the approximation at x
        return sum([C[n][1]*E**(1j*(-n)*w*(x+1)) for n in range(1,order+1)])+sum([C[n][0]*E**(1j*n*w*(x+1)) for n in range(1,order+1)])+C[0]
        #return sum([(np.real(C[n]) - np.imag(C[n])*1j)*E**(1j*(-n)*w*(x+1)) for n in range(1,order+1)])+sum([C[n]*E**(1j*n*w*(x+1)) for n in range(1,order+1)])+C[0]


#Real Fourier decomposition, discontinuous
class RealFourier(Extractor):
    def __init__(self):
        self.name = "RealFourier"

    def getC(self, n, X):
        #Use the data set to figure out omega and tau and such
        k = len(X) #Length of the set

        #Calculate and return the coefficients - bearing in mind the formula for the 0th is different
        if n == 0:
            return (1.0/k)*sum([X[j]+X[j-1] for j in range(1,k)])
        else:
            a = (1/(PI*n))*sum([(X[j]-X[j-1])*(np.sin((j-1)*2.0*PI*n/k)+(k/(2*PI*n))*(np.cos(2*PI*n*j/k)-np.cos(2*PI*n*(j-1)/k))) + X[j]*(np.sin(2*PI*n*j/k)-np.sin(2*PI*n*(j-1)/k)) for j in range(1,k)])
            b = (1/(PI*n))*sum([(X[j]-X[j-1])*(-np.cos((j-1)*2.0*PI*n/k)+(k/(2*PI*n))*(np.sin(2*PI*n*j/k)-np.sin(2*PI*n*(j-1)/k))) - X[j]*(np.cos(2*PI*n*j/k)-np.cos(2*PI*n*(j-1)/k)) for j in range(1,k)])
            return (a,b)

    def evalH(self, C, T, x):
        #Use T to calculate omega
        w = (2*PI)/T
        order = len(C)-1

        #Evaluate the approximation at x
        return (C[0]/2.0) + sum([C[n][0]*np.cos(n*w*x) + C[n][1]*np.sin(n*w*x) for n in range(1,order+1)])