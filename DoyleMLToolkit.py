#API for experimenting with orthogonal decomposition feature classification
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University
#The complete library can be found at www.github.com/spatrickdoyle/doyle-ml-toolkit

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import hashlib
import glob,os

TOKEN_PATH = "./decomps/"

#Tokens are passed data and generate and contain all the decomposed features of that data for a given method
class Token:
    def __init__(self, y, X, approximation, order, cost=False, zeroth=False):
        #list y[]: list of known classifications, may be empty
        #list X[dimension][row][order]: list of data sets
        #Approximation approximation: the feature coefficient object to use
        #int order: the number of coefficients to generate and use
        #bool cost: if true, keep track of the accuracy of the approximation being made
        #bool zeroth: if true, generate the 0th order coefficient

        self.dimension = len(X) #Dimension of data
        self.size = len(X[0]) #Number of rows
        self.length = len(X[0][0]) #Length of each row
        self.coefficients = []
        self.neg_coefs = []#This is where negative coefficients up to -order are stored
        self.X = X
        self.y = y
        self.method = approximation

        #Generate a token string using the data and metadata (different if different y)
        dataString = hashlib.md5(bytearray(str(y)+str(X))).hexdigest()
        methodString = hashlib.md5(bytearray(self.method.name+str(zeroth))).hexdigest()
        baseString = dataString+":"+methodString+":"
        self.string = baseString+str(order)

        #This is a multidimensional array: (self.weights[dimension][row][order])
        self.weights = []
        #Here's what it's for: if cost is true, each time an approximation is done, each time a coefficient
        #is generated, it uses costFunction to figure out how much the accuracy of the approximation
        #increases with the given term. It then divides each term by the total to create the weights

        #Check if there is an existing token file of the same data and method
        existingTokens = glob.glob(TOKEN_PATH+"%s:%s:*.csv"%(dataString,methodString))
        if len(existingTokens) > 0:
            #If there is, check to see if it is of the same or higher order
            orders = sorted([int(i.split(':')[-1][:-4]) for i in existingTokens])
            biggestString = baseString+str(orders[-1])
            if orders[-1] >= order:
                #If it is, load the correct order coefficients from that file
                tokenFile = file(TOKEN_PATH+biggestString+".csv","r")

                self.coefficients = [[] for i in range(self.dimension)]
                for line in tokenFile.readlines():
                    thisRow = line.split(',')
                    for i in range(self.dimension):
                        self.coefficients[i].append([])
                    for value in range(len(thisRow)):
                        if value == 0:
                            continue
                        elif len(thisRow[value]) == 0:
                            continue
                        else:
                            if thisRow[value] != '\n':
                                if (value-2)%(orders[-1]+2) < order+1:
                                    try:
                                        self.coefficients[(value-2)/(orders[-1]+2)][-1].append(float(thisRow[value]))
                                    except ValueError:
                                        self.coefficients[(value-2)/(orders[-1]+2)][-1].append(complex(thisRow[value]))
                tokenFile.close()
            else:
                #If not, load the partial file, generate the rest of the coefficients, and delete the old file

                #Load the existing file
                tokenFile = file(TOKEN_PATH+biggestString+".csv","r")

                self.coefficients = [[] for i in range(self.dimension)]
                for line in tokenFile.readlines():
                    thisRow = line.split(',')
                    for i in range(self.dimension):
                        self.coefficients[i].append([])
                    for value in range(len(thisRow)):
                        if value == 0:
                            continue
                        elif len(thisRow[value]) == 0:
                            continue
                        else:
                            if thisRow[value] != '\n':
                                try:
                                    self.coefficients[(value-2)/(orders[-1]+2)][-1].append(float(thisRow[value]))
                                except ValueError:
                                    self.coefficients[(value-2)/(orders[-1]+2)][-1].append(complex(thisRow[value]))
                tokenFile.close()

                #Generate new coefficients
                for dimension in range(self.dimension):
                    for row in range(self.size):
                        for n in range(orders[-1]+1,order+1):
                            self.coefficients[dimension][row].append(self.method.getC(n,X[dimension][row]))

                newTokenFile = file(TOKEN_PATH+self.string+".csv","w")

                for row in range(self.size):
                    tmp = []
                    tmp.append(self.y[row])
                    tmp.append('')
                    for dimension in range(self.dimension):
                        tmp += [str(i) for i in self.coefficients[dimension][row]]
                        tmp.append('')
                    newTokenFile.write(','.join(tmp)+'\n')

                newTokenFile.close()

                os.remove("%s"%(TOKEN_PATH+biggestString+".csv"))
        else:
            #If not, generate a new set of coefficients and write a new token file
            self.coefficients = [[] for i in range(len(self.X))]
            for dimension in range(len(X)):
                for row in range(len(X[dimension])):
                    self.coefficients[dimension].append([])
                    for n in range(order+1):
                        if (n == 0) and (zeroth == False):
                            self.coefficients[dimension][row].append(0)
                        else:
                            self.coefficients[dimension][row].append(self.method.getC(n,X[dimension][row]))

            tokenFile = file(TOKEN_PATH+self.string+".csv","w")

            for row in range(len(X[0])):
                tmp = []
                if len(self.y) > 0:
                    tmp.append(self.y[row])
                else:
                    tmp.append('')
                tmp.append('')
                for dimension in range(len(X)):
                    tmp += [str(i) for i in self.coefficients[dimension][row]]
                    tmp.append('')
                tokenFile.write(','.join(tmp)+'\n')

            tokenFile.close()

        #For now, negative coefficients will be generated each time no matter what and never stored in a file
        self.neg_coefs = [[] for i in range(len(self.X))]
        for dimension in range(len(X)):
            for row in range(len(X[dimension])):
                self.neg_coefs[dimension].append([])
                for n in range(1,order+1):
                    self.neg_coefs[dimension][row].append(self.method.getC(-n,X[dimension][row]))

        #If cost is set, calculate the weights
        if cost:
            costs = []
            for d in range(len(self.coefficients)):
                costs.append([])
                self.weights.append([])
                for r in range(len(self.coefficients[d])):
                    costs[-1].append([])
                    self.weights[-1].append([])
                    for n in range(len(self.coefficients[d][r])+1):
                        if n == 0:
                            h = lambda x: self.method.evalH([0],len(self.X[d][r]),x)
                        else:
                            h = lambda x: self.method.evalH(self.coefficients[d][r][:n],len(self.X[d][r]),x)
                        costs[-1][-1].append(self.costFunction(h,self.X[d][r]))
                    self.weights[-1][-1] = [abs(costs[-1][-1][i]-costs[-1][-1][i+1])/abs(costs[-1][-1][0]-costs[-1][-1][-1]) for i in range(len(costs[-1][-1])-1)]

    def __add__(self,other):
        #Token other: the token being added to the current one, must have same dimension
        #retuns a new Subtoken, consisting of the data from other appended to the data from self

        #Throw an error if they aren't the same dimension
        if other.dimension != self.dimension:
            print "Added Tokens must be the same dimension!"
            raise TypeError

        return Subtoken(self.y+other.y,[self.X[d]+other.X[d] for d in range(self.dimension)],[self.coefficients[d]+other.coefficients[d] for d in range(self.dimension)],[self.weights[d]+other.weights[d] for d in range(len(self.weights))])

    def genSubtoken(self, rows):
        #list rows[]: list of ints, the rows of coefficients to put in the subtoken
        #returns a Subtoken, which is like a token but has a different constructor and no associated file

        #Create a new list of lists of coefficients at the indices given by rows
        sorted_rows = sorted(rows)
        coefs = []
        for dimension in range(len(self.coefficients)):
            coefs.append([])
            for row in sorted_rows:
                coefs[dimension].append(self.coefficients[dimension][row])

        #Create a new list of classifications at the indices given by rows
        y = []
        for row in sorted_rows:
            y.append(self.y[row])

        #Create a new list of lists of data at the indices given by rows
        X = []
        for dimension in range(len(self.X)):
            X.append([])
            for row in sorted_rows:
                X[dimension].append(self.X[dimension][row])

        #Create a new list of lists of weights at the indices given by rows
        w = []
        for dimension in range(len(self.weights)):
            w.append([])
            for row in sorted_rows:
                w[dimension].append(self.weights[dimension][row])

        #Pass these lists to the Subtoken constructor
        new_subtoken = Subtoken(y,X,coefs,w)

        #Return the new Subtoken
        return new_subtoken

    def getBySweep(self, row, dimension=0):
        #int row: the index of the set of coefficients to return
        #int dimension: the data set to pull from
        #returns a list of ints, the coefficients of that sweep

        #Make a copy of the list of coefficients at index row
        row_copy = self.coefficients[dimension][row][:]

        #Return it
        return row_copy

    def getByOrder(self, order, dimension=0):
        #int order: the order of the set of coefficients to return
        #int dimension: the data set to pull from
        #returns a list of ints, the coefficients of that order for each sweep

        #Construct a list of coefficients at each index of the correct order
        coefs = []
        for row in self.coefficients[dimension]:
            coefs.append(row[order])

        #Return it
        return coefs

    def getAllCoefficients(self):
        #returns a copy of the whole coefficients matrix for this Token

        #Make a copy
        C_copy = self.coefficients[:]

        #Return it
        return C_copy

    def getAllY(self):
        #return the list of classifications provided to this token

        #Make a copy of this list so it can't get screwed up
        y_copy = self.y[:]

        #Return it
        return y_copy

    def getAllData(self):
        #return the matrix of data provided to this token

        #Make a copy of it so it can't get screwed up
        x_copy = self.X[:]

        #Return it
        return x_copy

    def getWeights(self, row, dimension=0):
        #int row: the row to return weights for
        #int dimension: the data set to pull from
        #return the list of weights associated with row

        #Make a copy
        w_copy = self.weights[dimension][row][:]

        #Return it
        return w_copy

    def getAllWeights(self):
        #int row: the row to return weights for
        #int dimension: the data set to pull from
        #return the list of weights associated with row

        #Make a copy
        w_copy = self.weights[:]

        #Return it
        return w_copy

    #This function is used internally
    def costFunction(self, h, y):
        #function h: the approximation to evaluate
        #list y[]: the list of points to check against
        #returns an int, the difference score

        return 0.5*sum([(np.real(h(i))-y[i])**2 for i in range(len(y))])

#A Subtoken does all the same things as a token, but is spawned by other Tokens and doesn't have a file
class Subtoken(Token):
    def __init__(self, y, X, C, w=[]):
        #list y[]: list of known classifications
        #list X[][][]: original data
        #list C[][][]: set of lists of coefficients
        #list w[][][]: set of lists of weights (may be empty)

        self.y = y
        self.X = X
        self.coefficients = C
        self.weights = w

        self.dimension = len(self.X)
        self.size = len(self.X[0])
        self.length = len(self.X[0][0])


#Orthogonal Decomposition Feature Classification
class ODFC:
    def __init__(self, approximation, selection, order, dimension=1):
        #class approximation: the feature coefficient function to use
        #class selection: the likelihood function class to use
        #int order: the number of coefficients to generate and use
        #int dimension: the number of vectors each data reading consists of

        self.C = approximation()
        self.dist = selection
        self.O = order
        self.D = dimension

        self.L = None

    def load(self, y, X, cost=False, zeroth=False):
        #list y[]: list of known classifications, may be empty
        #list X[][][]: list of data sets
        #OR
        #string y: path to .csv file with column list of known classifications, may be empty
        #string X: path to .csv file with matrix of data
        #bool cost: if True, it will generate the Token with weights
        #bool zeroth: if True, generate the 0th order coefficient
        #returns Token containing the features of the given data

        #If they are files, not lists...
        if type(X) is str:
            #Open the files
            if len(y) > 0:
                classFile = file(y,'r')
            else:
                classFile = -1
            dataFile = file(X,'r')

            dataLines = dataFile.readlines()
            dimMax = dataLines[0].split(',').count('')+1
            yMat = []
            XMat = [[] for i in range(dimMax)]

            #Assume the data in X is floats/complex and record it into a matrix
            for row in dataLines:
                if len(row) > 1:
                    dim = 0
                    for d in range(dimMax):
                        XMat[d].append([])
                    #if len(row)-1 == row.count(','):
                    #    continue
                    for value in row.split(','):
                        if value == "":
                            dim += 1
                        else:
                            try:
                                XMat[dim][-1].append(float(value))
                            except ValueError:
                                XMat[dim][-1].append(complex(value))

            #y can be empty for data that is going to be tested. If it is:
            if classFile == -1:
                yMat = []
            else:
                #y could potentially be extended to any data type, but for now assume floats unless the
                #elements are surrounded by quotes
                for value in classFile.readlines():
                    if (len(value) > 0) and (value != "\n"):
                        #Is it a string?
                        if (value[0:3] == '"""') and (value[-4:-1] == '"""'):
                            yMat.append(value[3:-4])
                        else:
                            yMat.append(float(value))

                classFile.close()
            dataFile.close()
        else:
            XMat = X
            yMat = y

        #Load and return the appropriate Token
        return Token(yMat,XMat,self.C,self.O,cost,zeroth)

    def train(self, data, exclude=[]):
        #Token data: Token of the data to use to train
        #list exclude[]: list of ints corresponding to the 0-indexed rows to ignore when training the model
        #returns Token containing only the excluded data

        if len(exclude) > 0:
            #Split the passed Token into usable and excluded data
            rows = [i for i in range(data.size) if i not in exclude]
            the_token = data.genSubtoken(rows)
            exclusion_token = data.genSubtoken(exclude)
        else:
            the_token = data
            exclusion_token = None

        #Use the data token to instantiate the appropriate distribution
        self.L = self.dist(the_token.getAllY(),the_token.getAllCoefficients(),the_token.getAllWeights())
        self.domain = len(the_token.getAllData()[0][0])

        #Return the exclusion token (might be None)
        return exclusion_token

    def predict(self, data, prediction=[]):
        #Token data: Token of the data to evaluate
        #list prediction[]: list of classifications to check the likelihood of
        #returns a list of tuples, each one corresponding to a row of the data, (most likely classification, normed likelihood of that classification) OR
        #if prediction is not empty, return a list of tuples, (classification from passed list, normed likelihood of that classification)

        #For each set of data in the Token...
        #If prediction is empty,
        ret = []
        if len(prediction) == 0:
            for row in range(len(data.getAllCoefficients()[0])):
                ret.append(self.L.mostLikely(data.getBySweep(row)))
        else:
            #If prediction is passed, then for each prediction
            for row in range(len(data.getAllCoefficients()[0])):
                ret.append((prediction[row],self.L.checkY(prediction[row],data.getBySweep(row))))

        #Return the constructed list
        return ret

    def test(self, data):
        #Token data: Token of data to test, must already contain classifications
        #returns an integer, the rate of correct identification

        if (len(data.getAllY()) != len(data.getAllData()[0])) or (data.getAllY().count("") != 0):
            print "Token must contain classifications"
            raise ValueError

        ret = []
        #For each set of data in the Token
        for row in range(len(data.getAllY())):
            #Retrain the model without the current row
            thisRow = self.train(data,[row])

            #Record the likelihood that the current row belongs to the known classification
            ret.append(self.predict(thisRow,thisRow.getAllY())[0][1])

        #Return the average
        return sum(ret)/len(ret)

    def genData(self, classification, n):
        #This is a public interface for the encapsulated distribution's genData method

        return self.L.genData(classification,n,self.domain)


    #These functions are for making plots and such
    def plotSample(self, data, row, color, imaginary=0, dimension=0):
        #Token data: Token storing the data to plot
        #int row: the data set to plot
        #string color: matplotlib color string, can be hex or just, like, 'red'
        #bool imaginary: 0 is plot real part, 1 plot imaginary part
        #int dimension: the data set to pull from
        #Plots the data with matplotlib, but plt.show() still needs to be run to display it

        theData = data.getAllData()
        if imaginary:
            plt.plot(range(len(theData[dimension][row])),[np.imag(i) for i in theData[dimension][row]],color)
        else:
            plt.plot(range(len(theData[dimension][row])),[np.real(i) for i in theData[dimension][row]],color)

    def plotApproximation(self, data, row, color, imaginary=0, dimension=0):
        #Token data: Token storing the approximation to plot
        #int row: the data set to plot
        #string color: matplotlib color string, can be hex or just, like, 'red'
        #bool imaginary: 0 is plot real part, 1 plot imaginary part
        #int dimension: the data set to pull from
        #Plots the approximation with matplotlib, but plt.show() still needs to be run to display it

        d = 1.0
        theData = data.getAllData()
        theCoefficients = data.getBySweep(row,dimension)

        #print theCoefficients[1:]
        #print [self.C.getC(n,theData[dimension][row]) for n in range(1,len(theCoefficients))]
        #print [self.C.getC(-n,theData[dimension][row]) for n in range(1,len(theCoefficients))]

        x = [i/d for i in range(int(d)*(len(theData[dimension][0])-1))]
        if imaginary:
            y = [np.imag(self.C.evalH(theCoefficients,len(theData[0][0]),i,data.neg_coefs[dimension][row])) for i in x]
        else:
            y = [np.real(self.C.evalH(theCoefficients,len(theData[0][0]),i,data.neg_coefs[dimension][row])) for i in x]
        plt.plot(x,y,color)

    def plotDistributionByOrder(self, n, dimension=0):
        #int n: the order of coefficient to show the distribution for
        #int dimension: the data set to pull from
        #Plots the distribution with matplotlib, but plt.show() still needs to be run to display it

        mC = self.L.getMeanFunc(n,dimension)
        vC = self.L.getVarianceFunc(n,dimension)
        if (len(mC) > 0) and (len(vC) > 0):
            [mC[i]*(x**i) for i in range(len(mC))]
            mX = []
            mY = []
            lX = []
            lY = []
            rX = []
            rY = []
        else:
            mC = self.L.getMeanVal(n,dimension)
            vC = self.L.getVarianceVal(n,dimension)

    def plotDistributionByClassification(self, c, color, dimension=0):
        #int c: the classification to show the distribution for
        #string color: matplotlib color string
        #int dimension: the data set to pull from
        #Plots the distribution with matplotlib, but plt.show() still needs to be run to display it

        if c not in self.L.classifications:
            print "%s is not a valid classifications..."%c
            raise ValueError

        theCoefficients = self.L.means[dimension][self.L.classifications.index(c)]
        x = [i/5.0 for i in range(5*100)]
        y = [self.C.evalH(theCoefficients,101,i) for i in x]
        plt.plot(x,y,color)


        new_color = colors.rgb_to_hsv(colors.hex2color(color))
        new_color[1] *= 0.67

        theCoefficientsL = [theCoefficients[i]+np.sqrt(self.L.variances[dimension][self.L.classifications.index(c)][i]) for i in range(len(theCoefficients))]
        x = [i/5.0 for i in range(5*100)]
        y = [self.C.evalH(theCoefficientsL,101,i) for i in x]
        plt.plot(x,y,color=colors.hsv_to_rgb(new_color))

        theCoefficientsR = [theCoefficients[i]-np.sqrt(self.L.variances[dimension][self.L.classifications.index(c)][i]) for i in range(len(theCoefficients))]
        x = [i/5.0 for i in range(5*100)]
        y = [self.C.evalH(theCoefficientsR,101,i) for i in x]
        plt.plot(x,y,color=colors.hsv_to_rgb(new_color))


        new_color[1] *= 0.5

        theCoefficientsL = [theCoefficients[i] + 2*np.sqrt(self.L.variances[dimension][self.L.classifications.index(c)][i]) for i in range(len(theCoefficients))]
        x = [i/5.0 for i in range(5*100)]
        y = [self.C.evalH(theCoefficientsL,101,i) for i in x]
        plt.plot(x,y,color=colors.hsv_to_rgb(new_color))

        theCoefficientsR = [theCoefficients[i] - 2*np.sqrt(self.L.variances[dimension][self.L.classifications.index(c)][i]) for i in range(len(theCoefficients))]
        x = [i/5.0 for i in range(5*100)]
        y = [self.C.evalH(theCoefficientsR,101,i) for i in x]
        plt.plot(x,y,color=colors.hsv_to_rgb(new_color))
