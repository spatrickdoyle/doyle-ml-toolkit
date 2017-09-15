#Collection of feature extraction functions
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import integrate,polyfit
from scipy.special import eval_chebyt
from numpy import polynomial
from scipy.special import binom

PI = np.pi
E = np.e

class Extractor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        #Set the string name of the method - that's it

        pass

    @abstractmethod
    def getC(n, X, order):
        #int n: the order of the coefficient to generate
        #list X[]: the data sweep being approximated
        #int order: order of the entire approximation
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

    def getC(self, n, X, order):
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


#Real Fourier decomposition, discontinuous
class RealFourier(Extractor):
    def __init__(self):
        self.name = "RealFourier"

    def getC(self, n, X, order):
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


#Use inscribed parabolas to smooth the function
class SmoothFourier(Extractor):
    def __init__(self):
        self.name = "SmoothFourier"

    def getC(self, n, X, order):
        #Calculate T and w
        #X = X_+[X_[0]]
        m = len(X)
        T = len(X)
        w = 2.0*PI/T
        d = 3

        #Calculate the polynomial regression of X of degree 10
        th = np.polyfit(range(m),X,d)

        if n == 0:
            return (2.0/T)*sum([th[d-j]*integrate.quad(lambda x: (x**j)*np.cos(n*w*x),0,T)[0] for j in range(d+1)])
        else:
            a = (2.0/T)*sum([th[d-j]*integrate.quad(lambda x: (x**j)*np.cos(n*w*x),0,T)[0] for j in range(d+1)])
            b = (2.0/T)*sum([th[d-j]*integrate.quad(lambda x: (x**j)*np.sin(n*w*x),0,T)[0] for j in range(d+1)])
            return (a,b)

    def evalH(self, C, T, x):
        #Use T to calculate omega
        w = (2*PI)/T
        order = len(C)-1

        #Evaluate the approximation at x
        return (C[0]/2.0) + sum([C[n][0]*np.cos(n*w*x) + C[n][1]*np.sin(n*w*x) for n in range(1,order+1)])


#Chebyshev approximation
class Chebyshev(Extractor):
    def __init__(self):
        self.name = "Chebyshev"

    def getC(self, n, X, order):
        N = 101.0
        #First and last term are halved for some reason, but I don't think it matters
        s = lambda x: sum([((X[j]-X[j-1])*(x-j)+X[j])*self.step(-(x-(j-1))*(x-j)) for j in range(1,len(X))])
        f = lambda x: s((len(X)-1)*x)/(len(X)-1)

        c = (2/N)*sum([f(np.cos((np.pi*(k-0.5))/N))*np.cos((np.pi*n*(k-0.5))/N) for k in range(1,int(N+1))])
        #print n,c
        return c

    def evalH(self, C, T, x):
        #Use T to calculate omega
        #w = (2*PI)/T
        order = len(C)-1
        x /= T-1

        #Evaluate the approximation at x
        return ((C[0]/2.0)+sum([C[k]*eval_chebyt(k,x) for k in range(1,order+1)]))*(T-1)

    #Heaviside step function
    def step(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        elif x == 0:
            return 0.5


#Bernstein approximation
class Bernstein(Extractor):
    def __init__(self):
        self.name = "Bernstein"
        self.choose = binom

    def getC(self, n, X, order):
        #First and last term are halved for some reason, but I don't think it matters
        s = lambda x: sum([((X[j]-X[j-1])*(x-j)+X[j])*self.step(-(x-(j-1))*(x-j)) for j in range(1,len(X))])
        f = lambda x: s((len(X)-1)*x)/(len(X)-1)

        c = f(float(n)/float(order))#*choose(order,n)
        return c

    def evalH(self, C, T, x):
        #Use T to calculate omega
        #w = (2*PI)/T
        order = len(C)-1
        x /= T-1

        #Evaluate the approximation at x
        return sum([C[v]*self.B(order,v)(x) for v in range(order)])*(T-1)

    #Heaviside step function
    def step(self, x):
        if x < 0:
            return 0
        elif x > 0:
            return 1
        elif x == 0:
            return 0.5

    def B(self, degree, i):
        coefficients = [0,]*i + [self.choose(degree, i)]
        first_term = polynomial.polynomial.Polynomial(coefficients)
        second_term = polynomial.polynomial.Polynomial([1,-1])**(degree - i)
        return first_term * second_term


#I don't think there's any point to this one
#Complex Fourier decomposition, forced continuous
'''class ContinuousFourier(Extractor):
    def __init__(self):
        self.name = "ContinuousFourier"

    def getC(self, n, X_, order):
        #Use the data set to figure out omega and tau and such
        X = X_+[X_[0]]
        m = len(X) #Length of the set
        T = len(X_) #Period of the set
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
        return sum([C[n][1]*E**(1j*(-n)*w*(x+1)) for n in range(1,order+1)])+sum([C[n][0]*E**(1j*n*w*(x+1)) for n in range(1,order+1)])+C[0]'''


#Polynomial regression - don't know if this is useful at all
'''class Taylor(Extractor):
    def __init__(self):
        self.name = "Taylor"

    def getC(self, n, X, order):
        theta = polyfit(range(len(X)),X,order)[::-1]

        return theta[n]

    def evalH(self, C, T, x):
        return sum([C[i]*x**i for i in range(len(C))])'''
