#API for experimenting with feature identification and machine learning
#Written by Sean Doyle in 2017 as part of research at Southern Methodist University
#The complete library can be found at www.github.com/spatrickdoyle/doyle-ml-toolkit

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import numpy as np
import hashlib
import glob,os

TOKEN_PATH = "./features/"

#Tokens are passed data and generate and contain all the decomposed features of that data for a given method
class Token:
    def __init__(self, y, X, method, order):
        #list y[]: list of known classifications, may be empty
        #list X[dimension][row][order]: list of data sets
        #Extractor method: the feature extraction object to use
        #int order: the number of features to generate and use

        self.order = order
        self.dimension = len(X) #Dimension of data
        self.size = len(X[0]) #Number of rows
        self.length = len(X[0][0]) #Length of each row
        self.features = []
        self.X = X
        self.y = y
        self.classifications = list(set(self.y)) #List of unique classifications
        self.method = method
        self.name = self.method.name
        self.feats = [] #Used by genVariances

        #Generate a token string using the data and metadata (different if different y)
        dataString = hashlib.md5(bytearray(str(y)+str(X))).hexdigest()
        methodString = hashlib.md5(bytearray(self.method.name)).hexdigest()
        baseString = dataString+"-"+methodString+"-"
        self.string = baseString+str(order)

        #Check if there is an existing token file of the same data and method
        existingTokens = glob.glob(TOKEN_PATH+"%s-%s-*.csv"%(dataString,methodString))
        if len(existingTokens) > 0:
            #If there is, check to see if it is of the same or higher order
            orders = sorted([int(i.split('-')[-1][:-4]) for i in existingTokens])
            biggestString = baseString+str(orders[-1])
            if orders[-1] >= order:
                #If it is, load the correct order features from that file
                tokenFile = file(TOKEN_PATH+biggestString+".csv","r")

                self.features = [[] for i in range(self.dimension)]
                for line in tokenFile.readlines():
                    thisRow = line.split(',')
                    for i in range(self.dimension):
                        self.features[i].append([])
                    for value in range(len(thisRow)):
                        if value == 0:
                            continue
                        elif len(thisRow[value]) == 0:
                            continue
                        else:
                            if thisRow[value] != '\n':
                                if (value-2)%(orders[-1]+2) < order+1:
                                    thisValue = thisRow[value].split(" ")
                                    if len(thisValue) > 1:
                                        try:
                                            first = float(thisValue[0])
                                        except ValueError:
                                            first = complex(thisValue[0])
                                        try:
                                            second = float(thisValue[1])
                                        except ValueError:
                                            second = complex(thisValue[1])
                                        thisValue = (first,second)
                                    else:
                                        try:
                                            thisValue = float(thisValue[0])
                                        except ValueError:
                                            thisValue = complex(thisValue[0])
                                    self.features[(value-2)/(orders[-1]+2)][-1].append(thisValue)
                tokenFile.close()
            else:
                #If not, load the partial file, generate the rest of the features, and delete the old file

                #Load the existing file
                tokenFile = file(TOKEN_PATH+biggestString+".csv","r")

                self.features = [[] for i in range(self.dimension)]
                for line in tokenFile.readlines():
                    thisRow = line.split(',')
                    for i in range(self.dimension):
                        self.features[i].append([])
                    for value in range(len(thisRow)):
                        if value == 0:
                            continue
                        elif len(thisRow[value]) == 0:
                            continue
                        else:
                            if thisRow[value] != '\n':
                                thisValue = thisRow[value].split(" ")
                                if len(thisValue) > 1:
                                    try:
                                        first = float(thisValue[0])
                                    except ValueError:
                                        first = complex(thisValue[0])
                                    try:
                                        second = float(thisValue[1])
                                    except ValueError:
                                        second = complex(thisValue[1])
                                    thisValue = (first,second)
                                else:
                                    try:
                                        thisValue = float(thisValue[0])
                                    except ValueError:
                                        thisValue = complex(thisValue[0])
                                self.features[(value-2)/(orders[-1]+2)][-1].append(thisValue)
                tokenFile.close()

                #Generate new features
                for dimension in range(self.dimension):
                    for row in range(self.size):
                        for n in range(orders[-1]+1,order+1):
                            self.features[dimension][row].append(self.method.getC(n,X[dimension][row],order))

                newTokenFile = file(TOKEN_PATH+self.string+".csv","w+")

                for row in range(self.size):
                    tmp = []
                    tmp.append(self.y[row])
                    tmp.append('')
                    for dimension in range(self.dimension):
                        tmp2 = []
                        for i in self.features[dimension][row]:
                            if type(i) is tuple:
                                tmp2.append(str(i[0])+" "+str(i[1]))
                            else:
                                tmp2.append(str(i))
                        tmp += tmp2
                        tmp.append('')
                    newTokenFile.write(','.join(tmp)+'\n')

                newTokenFile.close()

                os.remove("%s"%(TOKEN_PATH+biggestString+".csv"))
        else:
            #If not, generate a new set of features and write a new token file
            self.features = [[] for i in range(len(self.X))]
            for dimension in range(len(X)):
                for row in range(len(X[dimension])):
                    self.features[dimension].append([])
                    for n in range(order+1):
                        self.features[dimension][row].append(self.method.getC(n,X[dimension][row],order))

            tokenFile = file(TOKEN_PATH+self.string+".csv","w+")

            for row in range(len(X[0])):
                tmp = []
                if len(self.y) > 0:
                    tmp.append(self.y[row])
                else:
                    tmp.append('')
                tmp.append('')
                for dimension in range(len(X)):
                    tmp2 = []
                    for i in self.features[dimension][row]:
                        if type(i) is tuple:
                            tmp2.append(str(i[0])+" "+str(i[1]))
                        else:
                            tmp2.append(str(i))
                    tmp += tmp2
                    tmp.append('')
                tokenFile.write(','.join(tmp)+'\n')

            tokenFile.close()

    def __add__(self,other):
        #Token other: the token being added to the current one, must have same dimension
        #retuns a new Subtoken, consisting of the data from other appended to the data from self

        #Throw an error if they aren't the same dimension
        if other.dimension != self.dimension:
            print "Added Tokens must be the same dimension!"
            raise TypeError

        return Subtoken(self.y+other.y,[self.X[d]+other.X[d] for d in range(self.dimension)],[self.features[d]+other.features[d] for d in range(self.dimension)])

    def genSubtoken(self, rows):
        #list rows[]: list of ints, the rows of features to put into the subtoken
        #returns a Subtoken, which is like a token but has a different constructor and no associated file

        #Create a new list of lists of features at the indices given by rows
        sorted_rows = sorted(rows)
        feats = []
        for dimension in range(len(self.features)):
            feats.append([])
            for row in sorted_rows:
                feats[dimension].append(self.features[dimension][row])

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

        #Pass these lists to the Subtoken constructor
        new_subtoken = Subtoken(y,X,feats)

        #Return the new Subtoken
        return new_subtoken

    def genVariances(self, num_feats, imaginary=0, zeroth=1, dimension=0):
        #int num_feats: number of features to include in the self.feats array
        #int imaginary: if 0, use real part, if 1, use imaginary
        #bool zeroth: if true, use zeroth component
        #int dimension: dimension of the data to use
        #Generates a list self.feats of the indices of the n most significant features (as complex values)
        #returns a list of the percentage of the total variance accounted for by components generated

        im = imaginary

        imaginary = 0
        mu = [sum(self.getFeaturesByOrder(n,dimension,imaginary))/self.size for n in range((1-zeroth),self.order+1)]
        sigmasr = [(1.0/(self.size-1))*np.dot([j-mu[zeroth+n-1] for j in self.getFeaturesByOrder(n,dimension,imaginary)],[j-mu[zeroth+n-1] for j in self.getFeaturesByOrder(n,dimension,imaginary)]) for n in range((1-zeroth),self.order+1)]

        imaginary = 1
        mu = [sum(self.getFeaturesByOrder(n,dimension,imaginary))/self.size for n in range((1-zeroth),self.order+1)]
        sigmasi = [(1.0/(self.size-1))*np.dot([j-mu[zeroth+n-1] for j in self.getFeaturesByOrder(n,dimension,imaginary)],[j-mu[zeroth+n-1] for j in self.getFeaturesByOrder(n,dimension,imaginary)]) for n in range((1-zeroth),self.order+1)]

        percentagesr = np.asarray([i/sum(sigmasr) for i in sigmasr])
        percentagesi = np.asarray([i/sum(sigmasi) for i in sigmasi])

        self.feats = (np.argsort(-percentagesr)[:num_feats] + 1j*np.argsort(-percentagesi)[:num_feats]).tolist()

        if im == 0:
            return percentagesr.tolist()
        elif im == 1:
            return percentagesi.tolist()

    #Not sure this works at all correctly
    def getR2(self, row, imaginary=0, dimension=0):
        #int row: row to calculate r-squared value for
        #int imaginary: if 0, use real part, if 1, use imaginary
        #int dimension: dimension of the data to use
        #returns the r-squared score for the given sweep

        if imaginary == 0:
            data = np.real(self.X[dimension][row]).tolist()
            ybar = np.mean(data)
            SSres = sum([(data[i]-np.real(self.method.evalH(self.features[dimension][row],self.length,i,np.real(self.feats).astype(int))))**2 for i in range(self.length)])
        elif imaginary == 1:
            data = [np.imag(i) for i in self.X[dimension][row]]
            ybar = np.mean(data)
            SSres = sum([(data[i]-np.imag(self.method.evalH(self.features[dimension][row],self.length,i,np.imag(self.feats).astype(int))))**2 for i in range(self.length)])
        SStot = sum([(data[i]-ybar)**2 for i in range(self.length)])
        print SSres
        print SStot
        print SSres/SStot
        return 1.0-(SSres/SStot)

    def getFeaturesBySweep(self, row, dimension=0, part=-1, which=0):
        #int row: the index of the set of features to return
        #int dimension: the data set to pull from
        #int part: real (0) or imaginary (1) part, default is both (-1)
        #int which: which element of the tuple to return if it is tuples
        #returns a list of ints, the features of that sweep

        #Make a copy of the list of features at index row
        if part == -1:
            row_copy = [i[which] if type(i) is tuple else i for i in self.features[dimension][row]]
        elif part == 0:
            row_copy = [float(np.real(i[which])) if type(i) is tuple else float(np.real(i)) for i in self.features[dimension][row]]
        elif part == 1:
            row_copy = [float(np.imag(i[which])) if type(i) is tuple else float(np.imag(i)) for i in self.features[dimension][row]]

        #Return it
        return row_copy

    def getFeaturesByOrder(self, order, dimension=0, part=-1, which=0):
        #int order: the order of the set of features to return
        #int dimension: the data set to pull from
        #int part: real (0) or imaginary (1) part, default is both (-1)
        #int which: which element of the tuple to return if it is tuples
        #returns a list of ints, the features of that order for each sweep

        #Construct a list of features at each index of the correct order
        feats = []
        for row in self.features[dimension]:
            if type(row[order]) is tuple:
                if part == -1:
                    feats.append(row[order][which])
                elif part == 0:
                    feats.append(float(np.real(row[order][which])))
                elif part == 1:
                    feats.append(float(np.imag(row[order][which])))
            else:
                if part == -1:
                    feats.append(row[order])
                elif part == 0:
                    feats.append(float(np.real(row[order])))
                elif part == 1:
                    feats.append(float(np.imag(row[order])))

        #Return it
        return feats

    #This was cool and all, but it isn't actually returning a properly deep copy, so using it is probably bad practice. I'll phase it out eventually
    def getAllFeatures(self):
        #returns a copy of the whole feature matrix for this Token

        #Make a copy
        F_copy = self.features[:]

        #Return it
        return F_copy

    def getAllY(self):
        #return the list of classifications provided to this token

        #Make a copy of this list so it can't get screwed up
        y_copy = self.y[:]

        #Return it
        return y_copy

    #Same with this
    def getAllData(self):
        #return the matrix of data provided to this token

        #Make a copy of it so it can't get screwed up
        x_copy = self.X[:]

        #Return it
        return x_copy


#A Subtoken does all the same things as a token, but is spawned by other Tokens and doesn't have a file
class Subtoken(Token):
    def __init__(self, y, X, F):
        #list y[]: list of known classifications
        #list X[][][]: original data
        #list F[][][]: set of lists of features

        self.y = y
        self.X = X
        self.features = F

        self.dimension = len(self.X)
        self.size = len(self.X[0])
        self.length = len(self.X[0][0])
        self.order = len(F[0][0])-1


#Public interface for the toolkit
class Model:
    def __init__(self, extractor, selection, order, zeroth=1, dimension=1):
        #class extractor: the constructor of the feature extraction method to use
        #class selection: the likelihood function class to use
        #int order: the number of features to use
        #bool zeroth: whether the zeroth coefficient should be used to make classifications
        #int dimension: the number of vectors each data reading consists of

        self.zeroth = zeroth

        self.C = extractor()
        self.classifier = selection
        self.O = order
        self.D = dimension

        self.name = (self.C.name,self.classifier(None,0).name)

        self.L = None

    def load(self, y, X):
        #list y[]: list of known classifications, may be empty
        #list X[][][]: list of data sets
        #OR
        #string y: path to .csv file with column list of known classifications, may be empty
        #string X: path to .csv file with matrix of data
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
                        if value == "" or value == "\n":
                            dim += 1
                        else:
                            try:
                                XMat[dim][-1].append(float(value))
                            except ValueError:
                                value = value.replace("i","j")
                                if value[0] != '\xef':
                                    XMat[dim][-1].append(complex(value))
                                else:
                                    XMat[dim][-1].append(complex(value[3:]))

            #y can be empty for data that is going to be tested. If it is:
            if classFile == -1:
                yMat = []
            else:
                #y could potentially be extended to any data type, but for now assume floats unless the
                #elements are surrounded by quotes
                for value in classFile.readlines():
                    if (len(value) > 0) and (value != "\n"):
                        #Is it a string?
                        if value[0] == '\xef':
                            value = value[3:]
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
        if self.C.name == "PCA":
            self.C.load(XMat)
        return Token(yMat,XMat,self.C,self.O)

    def train(self, data, exclude=[], features=False, plot=False):
        #Token data: Token of the data to use to train
        #list exclude[]: list of ints corresponding to the 0-indexed rows to ignore when training the model
        #bool features: if True, use only the features from self.feats
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
        self.L = self.classifier(the_token,self.zeroth,data.feats,plot)
        self.domain = the_token.length

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
            for row in range(data.size):
                ret.append(self.L.mostLikely(data.getFeaturesBySweep(row)))
        else:
            #If prediction is passed, then for each prediction
            for row in range(data.size):
                ret.append((prediction[row],self.L.checkY(prediction[row],data.getFeaturesBySweep(row))))

        #Return the constructed list
        return ret

    def test(self, data, positive, features=[0,1], verbose=False, everything=False):
        #Token data: Token of data to test, must already contain classifications
        #string positive: the classification considered a 'positive' identification in the f1 calculation
        #bool verbose: if True, print each individual result
        #bool everything: if True, instead of summaries, return the result of every individual test
        #returns a list of floats, the percentage of correct identifications, the precision, the recall, the F1 score for the given class, using "leave one out" cross validation, and a nested list of the true negatives, false positives, false negatives, and true positives

        if (len(data.getAllY()) != data.size) or (data.getAllY().count("") != 0):
            print "Token must contain classifications"
            raise ValueError
        if positive not in data.getAllY():
            print "Positive value "+positive+" not a valid classification"
            raise SystemExit

        tp = []
        fp = []
        fn = []
        tn = []
        ret = 0.0
        results = []
        #For each set of data in the Token
        for row in range(len(data.getAllY())):
            #Retrain the model without the current row
            thisRow = self.train(data,[row],features)
            #temporarily, don't leave one out
            #self.train(data)
            #thisRow = data.genSubtoken([row])

            #Record the likelihood that the current row belongs to the known classification
            result = self.predict(thisRow,thisRow.getAllY())[0]
            results.append(result[1])
            if verbose:
                print row,result
            if (result[0] == positive) and (result[1] >= 0.5):
                tp.append(row)
            elif (result[0] != positive) and (result[1] < 0.5):
                fp.append(row)
            elif (result[0] == positive) and (result[1] < 0.5):
                fn.append(row)
            else:
                tn.append(row)

            #Count correct identifications
            if result[1] >= 0.5:
                ret += 1.0

        #Calculate precision and recall
        try:
            precision = float(len(tp))/(len(tp)+len(fp))
        except ZeroDivisionError:
            precision = -1
        try:
            recall = float(len(tp))/(len(tp)+len(fn))
        except ZeroDivisionError:
            recall = -1

        print len(tn),len(fp)
        print len(fn),len(tp)

        if everything == False:
            if precision == 0 or recall == 0:
                return [ret/data.size,precision,recall,-1,[tn,fp,fn,tp]]
            else:
                return [ret/data.size,precision,recall,2.0/((1.0/precision)+(1.0/recall)),[tn,fp,fn,tp]]
        else:
            return results

    #Not sure what to do with this until the new classifiers are figured out
    '''def genData(self, classification, n):
        #This is a public interface for the encapsulated distribution's genData method

        return self.L.genData(classification,n,self.domain)'''


    #These functions are for making plots and such
    def plotSample(self, data, row, color, imaginary=0, dimension=0):
        #Token data: Token storing the data to plot
        #int row: the data set to plot
        #string color: matplotlib color string, can be hex or just, like, 'red'
        #bool imaginary: 0 is plot real part, 1 plot imaginary part
        #int dimension: the data set to pull from
        #Plots the data with matplotlib, but plt.show() still needs to be run to display it

        theData = data.getAllData()
        X = np.logspace(4,8,data.length) # range(data.length)
        if imaginary:
            plt.plot(X,[np.imag(i) for i in theData[dimension][row]],color)
        else:
            plt.plot(X,[np.real(i) for i in theData[dimension][row]],color)

    def plotApproximation(self, data, row, color='', imaginary=0, dimension=0, feats=[]):
        #Token data: Token storing the approximation to plot
        #int row: the data set to plot
        #string color: matplotlib color string, can be hex or just, like, 'red'
        #bool imaginary: 0 is plot real part, 1 plot imaginary part
        #int dimension: the data set to pull from
        #Plots the approximation with matplotlib if color is not empty
        #Returns an array of the points being plotted

        d = 5.0#20.0
        theCoefficients = data.getAllFeatures()[dimension][row][:]
        first = theCoefficients[0]
        #if self.zeroth == 0:
        #    theCoefficients[0] = 0.0

        b1 = 0#-int(d*(data.length))
        b2 = int(d*(data.length-1))
        x = [i/d for i in range(b1,b2)]
        if imaginary:
            y = [np.imag(self.C.evalH(theCoefficients,data.length,i,feats)) for i in x]
        else:
            y = [np.real(self.C.evalH(theCoefficients,data.length,i,feats)) for i in x]
        x = np.logspace(4,8,len(y))
        if len(color) > 0:
            plt.plot(x,y,color)

        return [first]+[float(i) for i in y[1::int(d)]]

    def plotSamples(self, data, imaginary=0):
        #Token data: Token storing the data to plot
        #bool imaginary: 0 is plot real part, 1 plot imaginary part

        #Define the tags for the legend
        colors = ['red','cyan','green','purple','blue','orange']
        tags = [mpatches.Patch(color=colors[i], label=data.classifications[i]) for i in range(len(data.classifications))]

        plt.legend(handles=tags)
        for i in range(len(data.getAllY())):
                self.plotSample(data,i,colors[data.classifications.index(data.getAllY()[i])],imaginary)

    def plotApproximations(self, data, imaginary=0, feats=False):
        #Token data: Token storing the approximation to plot
        #bool imaginary: 0 is plot real part, 1 plot imaginary part

        #Define the tags for the legend
        colors = ['red','cyan','green','purple','blue','orange']
        tags = [mpatches.Patch(color=colors[i], label=data.classifications[i]) for i in range(len(data.classifications))]

        if feats:
            if imaginary == 0:
                feat = np.real(data.feats).astype(int)
            elif imaginary == 1:
                feat = np.imag(data.feats).astype(int)
        else:
            feat = []

        plt.legend(handles=tags)
        for i in range(len(data.getAllY())):
                self.plotApproximation(data,i,colors[data.classifications.index(data.getAllY()[i])],imaginary,0,feat)
