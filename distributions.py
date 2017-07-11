#Collection of functions which generate different kinds of likelihood functions
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University

from DoyleMLToolkit import Subtoken

from abc import ABCMeta, abstractmethod
from scipy.special import erf
import numpy as np

#Abstract distribution class
class Distribution:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, y, C, weights):
        #list y[]: corresponding list of classifications
        #list C[dimension][row][order]: nested list of coefficients to use to train the function

        #Weights may or may not be empty, and will only be used sometimes

        #self.classifications is a list of all possible classifications for the data. This is set in the
        #constructor if it is a discrete distribution, and left empty if continuous

        #Make list of lists of mean feature values for each order feature corresponding to the classifications
        #and list of lists of the variance values for each order's distribution corresponding to classifications
        #Both empty is continuous
        #self.means
        #self.variances

        #For a discrete distribution each one can be checked individually, but for
        #a continous one we gotta regress those values
        #These are lists of coefficient vectors for the mean, left and right variances for each order
        #self.thetaM = []
        #self.thetaL = []
        #self.thetaR = []

        pass

    @abstractmethod
    def mostLikely(self, c):
        #list c[]: list of coefficients representing the sweep to classify
        #returns a tuple, the most likely classification and its likelihood

        #For each order coefficient in c
        #If thetaM is empty, figure out which distribution on the plot gives the highest likelihood
        #If not, calculate the mean function value at that point

        #Use the given method to calculate the total normed likelihood

        #Return the tuple

        pass

    @abstractmethod
    def checkY(self, y, c):
        #y: classification to check (could be number or string depending on the data)
        #list c[]: list of coefficients representing the sweep to classify
        #returns the likelihood of the sweep represented by c being classified as y

        #If this is classification, not regression, make sure y is in self.classifications
        #If it isn't just return 0

        #Otherwise, figure out the total normed likelihood of that classification
        #This will be different for classification vs regression

        #Return it

        pass

    @abstractmethod
    def genData(self, classification, n):
        #classification: the classification the generated data should fall into - data type may vary
        #int n: the number of data points to generate
        #returns a list of floats, a new data set conforming to this distribution

        #If this is classification, not regression, make sure classification is in self.classifications
        #If it isn't just return 0

        #If this is classification, pull the mean and variance values directly for EACH order
        #If regression, pull the function coefficients and evaluate the function inverse to get the values

        #For each order, use numpy to generate a new value coefficient

        #Using that list of coefficients, generate an approximation
        #Evaluate it at points 0-n

        #Return that list

        pass


    #Accessors for the distribution representation functions
    def getMeanFunc(self, n, dimension=0):
        #int n: order of curve to return
        #int dimension: the data set to pull from
        #returns a list of the polynomial coefficients for the mean line

        #Make a copy
        tmp = self.thetaM[dimension][n][:]

        #Return it
        return tmp

    def getVarianceFunc(self, n, dimension=0):
        #int n: order of curve to return
        #int dimension: the data set to pull from
        #returns a tuple, the lists of the polynomial coefficients for the variance lines

        #Make a copy
        tmpL = self.thetaL[dimension][n][:]
        tmpR = self.thetaR[dimension][n][:]

        #Return it
        return (tmpL,tmpR)

    def getMeanVal(self, n, classification, dimension=0):
        #int n: the order of points to use
        #classification: the classification to pull from, data type is variant
        #int dimension: the data set to pull from

        return self.means[dimension][self.classifications.index(classification)][n]

    def getVarianceVal(self, n, classification, dimension=0):
        #int n: the order of points to use
        #classification: the classification to pull from, data type is variant
        #int dimension: the data set to pull from

        return self.variances[dimension][self.classifications.index(classification)][n]


#Discrete classification, assumes that the distribution of points is Gaussian, and
#evaluates each order of feature separately and just averages them without weights
#Data must be single-dimensional for now, but can be complex
#Also uses POPULATION VARIANCE, rather than sample variance
class UnweightedGaussianClassification(Distribution):
    def __init__(self, y, C, weights):
        #list y[]: corresponding list of classifications
        #list C[dimension][row][order]: nested list of coefficients to use to train the function

        #Weights may or may not be empty, and will only be used sometimes
        self.weights = weights

        #List of all possible classifications for the data. This is set in the
        #constructor if it is a discrete distribution, and left empty if continuous
        if type(y[0]) is str:
            self.discrete = True
        else:
            self.discrete = False

        if self.discrete:
            self.classifications = list(set(y))
        else:
            self.classifications = []

        #Make list of lists of mean feature values for each order feature corresponding to the classifications
        #and list of lists of the variance values for each order's distribution corresponding to classifications
        #Both empty is continuous
        self.means = []
        self.variances = []
        if self.discrete:
            #self.means[dimension][classification][order]
            self.means = [[[0 for j in C[0][0]] for i in self.classifications] for k in C]
            #self.variances[dimension][classification][order]
            self.variances = [[[0 for j in C[0][0]] for i in self.classifications] for k in C]

            self.samples = [[[[] for j in C[0][0]] for i in self.classifications] for k in C]

            for d in range(len(C)):
                for row in range(len(y)):
                    for n in range(len(C[0][0])):
                        self.samples[d][self.classifications.index(y[row])][n].append(C[d][row][n])
            for d in range(len(C)):
                for classification in range(len(self.samples[0])):
                    for n in range(len(C[0][0])):
                        self.means[d][classification][n] = sum(self.samples[d][classification][n])/len(self.samples[d][classification][n])
                        self.variances[d][classification][n] = sum([(np.real(i-self.means[d][classification][n])**2) + (np.imag(i-self.means[d][classification][n])**2)*1j for i in self.samples[d][classification][n]])/len(self.samples[d][classification][n])
        else:
            #For a discrete distribution each one can be checked individually, but for
            #a continous one we gotta regress those values
            #These are lists of coefficient vectors for the mean, left and right variances for each order
            self.thetaM = []
            self.thetaL = []
            self.thetaR = []

    def mostLikely(self, c):
        #list c[]: list of coefficients representing the sweep to classify
        #returns a tuple, the most likely classification and its likelihood

        #If not discrete, throw and error
        if len(self.means) == 0:
            print "Data needs to be discrete classification!"
            raise ValueError

        probs = []#probs[classification][order]

        for cls in range(len(self.means[0])):
            probs.append([])
            #For each order coefficient in c
            for n in range(len(c)):
                #Figure out the likelihood of the given coefficient falling in the distribution of this class
                prbreal = (np.e**(-(np.real(c[n]-self.means[0][cls][n])**2)/(2.0*np.real(self.variances[0][cls][n]))))/np.sqrt(2.0*np.pi*np.real(self.variances[0][cls][n]))
                prbimag = (np.e**(-(np.imag(c[n]-self.means[0][cls][n])**2)/(2.0*np.imag(self.variances[0][cls][n]))))/np.sqrt(2.0*np.pi*np.imag(self.variances[0][cls][n]))
                if np.isnan(prbimag):
                    prbimag = 1

                #Add it to the list
                probs[-1].append(prbreal*prbimag)

        #Use the given method to calculate the total normed likelihood
        probs = [sum(i)/len(i) for i in probs]

        #Return the tuple
        mx = max(probs)
        return (self.classifications[probs.index(mx)],mx/sum(probs))

    def checkY(self, y, c):
        #y: classification to check (could be number or string depending on the data)
        #list c[]: list of coefficients representing the sweep to classify
        #returns the likelihood of the sweep represented by c being classified as y

        #Make sure y is in self.classifications
        if y not in self.classifications:
            #If it isn't just return 0
            print "%s is not one of the possible classifications..."
            return 0

        #Otherwise, figure out the total normed likelihood of that classification
        probs = []#probs[classification][order]

        for cls in range(len(self.means[0])):
            probs.append([])
            #For each order coefficient in c
            for n in range(len(c)):
                #Figure out the likelihood of the given coefficient falling in the distribution of this class
                prbreal = (np.e**(-(np.real(c[n]-self.means[0][cls][n])**2)/(2.0*np.real(self.variances[0][cls][n]))))/np.sqrt(2.0*np.pi*np.real(self.variances[0][cls][n]))
                prbimag = (np.e**(-(np.imag(c[n]-self.means[0][cls][n])**2)/(2.0*np.imag(self.variances[0][cls][n]))))/np.sqrt(2.0*np.pi*np.imag(self.variances[0][cls][n]))
                if np.isnan(prbimag):
                    prbimag = 1

                #Add it to the list
                probs[-1].append(prbreal*prbimag)

        #Use the given method to calculate the total normed likelihood
        probs = [sum(i)/len(i) for i in probs]

        #Return it
        return probs[self.classifications.index(y)]/sum(probs)

    def genData(self, classification, n, domain):
        #classification: the classification the generated data should fall into - data type may vary
        #int n: the number of data points to generate
        #returns a Token, a new data set conforming to this distribution

        #Make sure y is in self.classifications
        if classification not in self.classifications:
            #If it isn't just return 0
            print "%s is not one of the possible classifications..."
            return 0

        newCoefs = [[] for i in range(n)]
        #For each order, use numpy to generate a new value coefficient
        for order in range(len(self.means[0][0])):
            new = np.random.normal(self.means[0][self.classifications.index(classification)][order],np.sqrt(self.variances[0][self.classifications.index(classification)][order]),n)
            for i in range(len(new)):
                newCoefs[i].append(new[i])

        #Return a new Token object
        return Subtoken([classification for i in range(n)],[[[0 for i in range(domain)]]],[newCoefs])
