#Collection of functions which generate different kinds of likelihood functions
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University

from DoyleMLToolkit import Subtoken
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB

from abc import ABCMeta, abstractmethod
import numpy as np

from scipy.stats import gaussian_kde


#Abstract distribution class
class Classifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, data, zeroth, features=[], plot=False):
        #Token data: Token containing the features and classifications to use
        #bool zeroth: if True, use zeroth coefficient

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


#RIGHT NOW THIS USES THE LAST 2 COEFFICIENTS GENERATED TO MAKE THE DECISION
class NaiveBayesReal(Classifier):
    def __init__(self, data, zeroth, features=[], plot=False):
        self.name = "Naive Bayes Real"
        self.zeroth = zeroth
        self.y = data.getAllY()

        self.classes = sorted(list(set(self.y)))
        self.X = [[data.getFeaturesBySweep(row,0,0)[features[0]],data.getFeaturesBySweep(row,0,0)[features[1]]] for row in range(data.size)]

        self.gnb = GaussianNB()
        self.trained = self.gnb.fit(self.X,self.y)

        self.features = features

    def mostLikely(self, c):
        x = [[float(np.real(j)) for j in [c[self.features[0]],c[self.features[1]]]]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return (classifications[0],probs[0][self.classes.index(classifications[0])])

    def checkY(self, y, c):
        x = [[float(np.real(j)) for j in [c[self.features[0]],c[self.features[1]]]]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return probs[0][self.classes.index(y)]

    def genData(self):
        return -1

class NaiveBayesImag(Classifier):
    def __init__(self, data, zeroth, features=[], plot=False):
        self.name = "Naive Bayes Imaginary"
        self.zeroth = zeroth
        self.y = data.getAllY()

        self.classes = sorted(list(set(self.y)))
        self.X = [[data.getFeaturesBySweep(row,0,1)[features[0]],data.getFeaturesBySweep(row,0,1)[features[1]]] for row in range(data.size)]

        self.gnb = GaussianNB()
        self.trained = self.gnb.fit(self.X,self.y)

        self.features = features

    def mostLikely(self, c):
        x = [[float(np.imag(j)) for j in [c[self.features[0]],c[self.features[1]]]]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return (classifications[0],probs[0][self.classes.index(classifications[0])])

    def checkY(self, y, c):
        x = [[float(np.imag(j)) for j in [c[self.features[0]],c[self.features[1]]]]]
        classifications = self.trained.predict(x)
        probs = self.trained.predict_proba(x)
        return probs[0][self.classes.index(y)]

    def genData(self):
        return -1


class NBKernelReal(Classifier):
    def __init__(self, data, zeroth, features=[], plot=False):
        self.name = "Kernel Density Estimation Real"
        self.zeroth = zeroth
        self.y = data.getAllY()
        self.unique = list(set(self.y))

        self.classes = sorted(list(set(self.y)))
        self.X = [data.getFeaturesBySweep(row,0,0) for row in range(data.size)]

        self.features = features

        self.classifiers = []
        for cls in self.unique:
            X = [[],[]]
            for sweep in range(len(self.y)):
                if self.y[sweep] != cls: continue
                X[0].append(self.X[sweep][features[0]])
                X[1].append(self.X[sweep][features[1]])
            self.classifiers.append(gaussian_kde(X))

            if plot:
                xmin = min(X[0])
                xmax = max(X[0])
                ymin = min(X[1])
                ymax = max(X[1])
                Xx, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([Xx.ravel(), Y.ravel()])
                values = np.vstack([X[0], X[1]])
                Z = np.reshape(self.classifiers[-1](positions).T, Xx.shape)

                fig,ax = plt.subplots()
                ax.imshow(np.rot90(Z),cmap=plt.cm.viridis,extent=[xmin, xmax, ymin, ymax])
                ax.plot(X[0],X[1],'ko')
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.set_aspect('auto')
                ax.set_title(cls)
                plt.show()

    def mostLikely(self, c):
        x = [float(np.real(j)) for j in c]
        #classifications = self.nbk.predict(x).tolist()
        probs = [i.pdf([[x[self.features[0]]],[x[self.features[1]]]]).tolist()[0] for i in self.classifiers] #List of probabilities c belongs to each class in self.unique
        probs = [i if not np.isinf(i) else float(1.797693134e308) for i in probs]
        probs_cpy = probs[:]
        maxx = max(probs)
        probs.remove(maxx)
        return (self.unique[probs_cpy.index(maxx)],maxx/(maxx+max(probs)))

    def checkY(self, y, c):
        x = [float(np.real(j)) for j in c]
        #classifications = self.nbk.predict(x).tolist()
        probs = [i.pdf([[x[self.features[0]]],[x[self.features[1]]]]).tolist()[0] for i in self.classifiers] #List of probabilities c belongs to each class in self.unique
        probs = [i if not np.isinf(i) else float(1.797693134e308) for i in probs]
        probs_cpy = probs[:]
        maxx = max(probs)
        probs.remove(maxx)
        return probs_cpy[self.unique.index(y)]/(maxx+max(probs))

    def genData(self):
        return -1

class NBKernelImag(Classifier):
    def __init__(self, data, zeroth, features=[], plot=False):
        self.name = "Kernel Density Estimation Imaginary"
        self.zeroth = zeroth
        self.y = data.getAllY()
        self.unique = list(set(self.y))

        self.classes = sorted(list(set(self.y)))
        self.X = [data.getFeaturesBySweep(row,0,1) for row in range(data.size)]

        self.features = features

        self.classifiers = []
        for cls in self.unique:
            X = [[],[]]
            for sweep in range(len(self.y)):
                if self.y[sweep] != cls: continue
                X[0].append(self.X[sweep][features[0]])
                X[1].append(self.X[sweep][features[1]])
            self.classifiers.append(gaussian_kde(X))

            if plot:
                xmin = min(X[0])
                xmax = max(X[0])
                ymin = min(X[1])
                ymax = max(X[1])
                Xx, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([Xx.ravel(), Y.ravel()])
                values = np.vstack([X[0], X[1]])
                Z = np.reshape(self.classifiers[-1](positions).T, Xx.shape)

                fig,ax = plt.subplots()
                ax.imshow(np.rot90(Z),cmap=plt.cm.viridis,extent=[xmin, xmax, ymin, ymax])
                ax.plot(X[0],X[1],'ko')
                ax.set_xlim([xmin,xmax])
                ax.set_ylim([ymin,ymax])
                ax.set_aspect('auto')
                ax.set_title(cls)
                plt.show()

    def mostLikely(self, c):
        x = [float(np.imag(j)) for j in c]
        #classifications = self.nbk.predict(x).tolist()
        probs = [i.pdf([[x[self.features[0]]],[x[self.features[1]]]]).tolist()[0] for i in self.classifiers] #List of probabilities c belongs to each class in self.unique
        probs = [i if not np.isinf(i) else float(1.797693134e308) for i in probs]
        probs_cpy = probs[:]
        maxx = max(probs)
        probs.remove(maxx)
        return (self.unique[probs_cpy.index(maxx)],maxx/(maxx+max(probs)))

    def checkY(self, y, c):
        x = [float(np.imag(j)) for j in c]
        #classifications = self.nbk.predict(x).tolist()
        probs = [i.pdf([[x[self.features[0]]],[x[self.features[1]]]]).tolist()[0] for i in self.classifiers] #List of probabilities c belongs to each class in self.unique
        probs = [i if not np.isinf(i) else float(1.797693134e308) for i in probs]
        probs_cpy = probs[:]
        maxx = max(probs)
        probs.remove(maxx)
        return probs_cpy[self.unique.index(y)]/(maxx+max(probs))

    def genData(self):
        return -1
