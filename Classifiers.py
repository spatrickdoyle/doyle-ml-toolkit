#Collection of functions which generate different kinds of likelihood functions
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University

from DoyleMLToolkit import Subtoken
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from abc import ABCMeta, abstractmethod
from scipy.special import erf
import numpy as np
import copy

#Abstract distribution class
class Classifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, y, C, zeroth, avg=[], method=None):
        #list y[]: corresponding list of classifications
        #list C[dimension][row][order]: nested list of coefficients to use to train the function
        #list avg[][][]: list of average values for the data sweeps [dimension][classification][value], may be empty

        #If avg is populated, use weights. If not, ignore it

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


    #This function is used internally
    def costFunction(self, h, y):
        #function h: the approximation to evaluate
        #list y[]: the list of points to check against
        #returns an int, the difference score

        return 0.5*sum([(np.real(h(i)-y[i]))**2 for i in range(len(y))]) + 0.5*sum([(np.imag(h(i)-y[i]))**2 for i in range(len(y))])


#Discrete classification, assumes that the distribution of points is Gaussian, and
#evaluates each order of feature separately and just averages them with or without weights
#Data must be single-dimensional for now, but can be complex
#Also uses POPULATION VARIANCE, rather than sample variance
class GaussianClassification(Classifier):
    def __init__(self, y, C, zeroth, avg=[], method=None):
        #list y[]: corresponding list of classifications
        #list C[dimension][row][order]: nested list of coefficients to use to train the function
        #bool zeroth: whether the zeroth coefficient should be used when making classifications
        #list avg[][][]: list of average values for the data sweeps [dimension][classification][value], may be empty
        #function method: instance of evalH function to use with avg to calculate differences

        self.zeroth = 1-int(zeroth)

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
            self.means = [[[[0,0] for j in C[0][0]] for i in self.classifications] for k in C]
            #self.variances[dimension][classification][order]
            self.variances = [[[[0,0] for j in C[0][0]] for i in self.classifications] for k in C]

            self.samples = [[[[] for j in C[0][0]] for i in self.classifications] for k in C]

            for d in range(len(C)):
                for row in range(len(y)):
                    for n in range(len(C[0][0])):
                        self.samples[d][self.classifications.index(y[row])][n].append(C[d][row][n])
            for d in range(len(C)):
                for classification in range(len(self.samples[0])):
                    for n in range(len(C[0][0])):
                        if type(self.samples[d][classification][n][0]) is tuple:
                            for i in range(2):
                                self.means[d][classification][n][i] = sum([j[i] for j in self.samples[d][classification][n]])/len(self.samples[d][classification][n])
                                self.variances[d][classification][n][i] = sum([(np.real(k[i]-self.means[d][classification][n][i])**2) + (np.imag(k[i]-self.means[d][classification][n][i])**2)*1j for k in self.samples[d][classification][n]])/len(self.samples[d][classification][n])
                        else:
                            self.means[d][classification][n] = sum(self.samples[d][classification][n])/len(self.samples[d][classification][n])
                            self.variances[d][classification][n] = sum([(np.real(k-self.means[d][classification][n])**2) + (np.imag(k-self.means[d][classification][n])**2)*1j for k in self.samples[d][classification][n]])/len(self.samples[d][classification][n])
        else:
            #For a discrete distribution each one can be checked individually, but for
            #a continous one we gotta regress those values
            #These are lists of coefficient vectors for the mean, left and right variances for each order
            self.thetaM = []
            self.thetaL = []
            self.thetaR = []

        self.average = avg
        self.evalH = method


        #ASDFASDF
        #d = 1.0
        #x = [i/d for i in range(int(d*(len(self.average[0][0])-1)))]
        #ASDFASDF

        if (len(self.average) > 0) and (self.evalH != None):
            self.weights = [[[[0,0] for j in C[0][0]] for i in self.classifications] for k in C]
            sums = [[0 for i in self.classifications] for k in C]

            for d in range(len(self.means)):
                for classification in range(len(self.classifications)):
                    minimum = self.costFunction(lambda x:self.evalH(self.means[d][classification],x),self.average[d][classification])
                    for n in range(self.zeroth,len(self.means[d][classification])):
                        if type(self.means[d][classification][n]) is list:
                            for i in range(2):
                                current = copy.deepcopy(self.means[d][classification])
                                current[n][i] = 0
                                #y = [np.imag(self.evalH(current,ii)) for ii in x]
                                #print current
                                #plt.plot(x,y,'#ff0000')
                                #plt.show()
                                score = self.costFunction(lambda x:self.evalH(current,x),self.average[d][classification])
                                self.weights[d][classification][n][i] = score-minimum
                                sums[d][classification] += self.weights[d][classification][n][i]
                        else:
                            current = copy.deepcopy(self.means[d][classification])
                            current[n] = 0
                            score = self.costFunction(lambda x:self.evalH(current,x),self.average[d][classification])
                            self.weights[d][classification][n] = score-minimum
                            sums[d][classification] += self.weights[d][classification][n]
                    for n in range(self.zeroth,len(self.means[d][classification])):
                        if type(self.means[d][classification][n]) is list:
                            for i in range(2):
                                self.weights[d][classification][n][i] /= sums[d][classification]
                        else:
                            self.weights[d][classification][n] /= sums[d][classification]
                    #print self.weights[d][classification]

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
            for n in range(self.zeroth,len(c)):
                if type(c[n]) is tuple:
                    for i in range(2):
                        #Figure out the likelihood of the given coefficient falling in the distribution of this class
                        prb = self.likelihood(c[n][i],self.means[0][cls][n][i],self.variances[0][cls][n][i])
                        if len(self.average) == 0:
                            probs[-1].append(prb)
                        else:
                            probs[-1].append(self.weights[0][cls][n][i]*prb)
                else:
                    prb = self.likelihood(c[n],self.means[0][cls][n],self.variances[0][cls][n])
                    if len(self.average) == 0:
                        probs[-1].append(prb)
                    else:
                        probs[-1].append(self.weights[0][cls][n]*prb)

        #Use the given method to calculate the total normed likelihood
        if len(self.average) == 0:
            probs = [sum(i)/len(i) for i in probs]
        else:
            probs = [sum(i) for i in probs]

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
            for n in range(self.zeroth,len(c)):
                if type(c[n]) is tuple:
                    for i in range(2):
                        #Figure out the likelihood of the given coefficient falling in the distribution of this class
                        prb = self.likelihood(c[n][i],self.means[0][cls][n][i],self.variances[0][cls][n][i])
                        if len(self.average) == 0:
                            probs[-1].append(prb)
                        else:
                            probs[-1].append(self.weights[0][cls][n][i]*prb)
                else:
                    prb = self.likelihood(c[n],self.means[0][cls][n],self.variances[0][cls][n])
                    if len(self.average) == 0:
                        probs[-1].append(prb)
                    else:
                        probs[-1].append(self.weights[0][cls][n]*prb)

        #Use the given method to calculate the total normed likelihood
        if len(self.average) == 0:
            probs = [sum(i)/len(i) for i in probs]
        else:
            probs = [sum(i) for i in probs]

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
            if type(self.means[0][self.classifications.index(classification)][order]) is list:
                new1 = np.random.normal(self.means[0][self.classifications.index(classification)][order][0], np.sqrt(self.variances[0][self.classifications.index(classification)][order][0]),n)
                new2 = np.random.normal(self.means[0][self.classifications.index(classification)][order][1], np.sqrt(self.variances[0][self.classifications.index(classification)][order][1]),n)
                new = [[new1[ii],new2[ii]] for ii in range(n)]
            else:
                new = np.random.normal(self.means[0][self.classifications.index(classification)][order],np.sqrt(self.variances[0][self.classifications.index(classification)][order]),n)
            for i in range(len(new)):
                newCoefs[i].append(new[i])

        #Return a new Token object
        return Subtoken([classification for i in range(n)],[[[0 for i in range(domain)] for i in range(n)]],[newCoefs])

    def likelihood(self, x, mu, sigma):
        prbreal = (np.e**(-(np.real(x-mu)**2)/(2.0*np.real(sigma))))/np.sqrt(2.0*np.pi*np.real(sigma))
        prbimag = (np.e**(-(np.imag(x-mu)**2)/(2.0*np.imag(sigma))))/np.sqrt(2.0*np.pi*np.imag(sigma))
        if np.isnan(prbreal):
            prbreal = 1
        if np.isnan(prbimag):
            prbimag = 1
        return prbreal*prbimag


class NaiveBayesReal(Classifier):
    def __init__(self, y, C, zeroth, avg=[], method=None):
        self.y = y
        self.C = C

        self.classes = sorted(list(set(self.y)))
        self.X = [[float(np.real(j[0])) if type(j) is tuple else float(np.real(j)) for j in i] for i in C[0]]

        self.gnb = GaussianNB()
        self.trained = self.gnb.fit(self.X,self.y)

        #classifications = self.trained.predict(self.X)
        #probs = self.trained.predict_proba(self.X)
        #print len(classifications[0]),len(probs[0])
        #for i in range(len(classifications)):
        #    print classifications[i],probs[i][self.classes.index(classifications[i])]/sum(probs[i])

    def mostLikely(self, c):
        x = [[float(np.real(j[0])) if type(j) is tuple else float(np.real(j)) for j in c]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return (classifications[0],probs[0][self.classes.index(classifications[0])]/sum(probs[0]))

    def checkY(self, y, c):
        x = [[float(np.real(j[0])) if type(j) is tuple else float(np.real(j)) for j in c]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return probs[0][self.classes.index(y)]/sum(probs[0])

    def genData(self):
        return -1

class NaiveBayesImag(Classifier):
    def __init__(self, y, C, zeroth, avg=[], method=None):
        self.y = y
        self.C = C

        self.classes = sorted(list(set(self.y)))
        self.X = [[float(np.imag(j[0])) if type(j) is tuple else float(np.imag(j)) for j in i] for i in C[0]]

        self.gnb = GaussianNB()
        self.trained = self.gnb.fit(self.X,self.y)

        #classifications = self.trained.predict(self.X)
        #probs = self.trained.predict_proba(self.X)
        #print len(classifications[0]),len(probs[0])
        #for i in range(len(classifications)):
        #    print classifications[i],probs[i][self.classes.index(classifications[i])]/sum(probs[i])

    def mostLikely(self, c):
        x = [[float(np.imag(j[0])) if type(j) is tuple else float(np.imag(j)) for j in c]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return (classifications[0],probs[0][self.classes.index(classifications[0])]/sum(probs[0]))

    def checkY(self, y, c):
        x = [[float(np.imag(j[0])) if type(j) is tuple else float(np.imag(j)) for j in c]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return probs[0][self.classes.index(y)]/sum(probs[0])

    def genData(self):
        return -1
