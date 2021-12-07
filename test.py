# -*- coding: utf-8 -*-
"""
Author: LOL group     
"""

""" =======================  Import dependencies ========================== """
import cv2, math, operator, warnings, os, timeit
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from itertools import combinations 
from abc import ABCMeta, abstractmethod
warnings.simplefilter("ignore") #Ignore all warning messages.
plt.close('all') #close any open plots

""" ======================  Function definitions ========================== """

''' define EHC '''
class DescriptorComputer: #Object definition based on abstract concept.
    __metaclass__ = ABCMeta
	
#   @abstractmethod
    def compute(self, frame):
        pass

class EdgeHistogramComputer(DescriptorComputer): #Decompose the picture into features by EHC method.

    def __init__(self, rows, cols):
        sqrt2 = math.sqrt(2)		
        self.kernels = (np.matrix([[1,1],[-1,-1]]), \
                        np.matrix([[1,-1],[1,-1]]),         \
                        np.matrix([[sqrt2,0],[0,-sqrt2]]),  \
                        np.matrix([[0,sqrt2],[-sqrt2,0]]),  \
                        np.matrix([[2,-2],[-2,2]]));
        self.bins = [len(self.kernels)]
        self.range = [0,len(self.kernels)]
        self.rows = rows
        self.cols = cols
        self.prefix = "EDH"
            
    def compute(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        descriptor = []
        dominantGradients = np.zeros_like(frame)
        maxGradient = cv2.filter2D(frame, cv2.CV_32F, self.kernels[0])
        maxGradient = np.absolute(maxGradient)
        for k in range(1,len(self.kernels)):
            kernel = self.kernels[k]
            gradient = cv2.filter2D(frame, cv2.CV_32F, kernel)
            gradient = np.absolute(gradient)
            np.maximum(maxGradient, gradient, maxGradient)
            indices = (maxGradient == gradient)
            dominantGradients[indices] = k
        #print(dominantGradients.shape)
        frameH, frameW = frame.shape
        for row in range(self.rows):
            for col in range(self.cols):
                mask = np.zeros_like(frame)
                mask[((frameH//self.rows)*row):((frameH//self.rows)*(row+1)),(frameW//self.cols)*col:((frameW//self.cols)*(col+1))] = 255
                hist = cv2.calcHist([dominantGradients], [0], mask, self.bins, self.range)
                #hist = cv2.normalize(hist, None)   #normalize
                descriptor.append(hist)
        return np.concatenate([x for x in descriptor])

def PGM_Parameters(FeatureExtractions, Labels): #Find the PGM parameters inlcuding mean, covariance, and prior probability.
    muSet = {}
    covSet = {}
    pcSet = {}
    classes = np.unique(Labels)
    for c in classes:
        EachFeatureSet = FeatureExtractions[Labels==c]
        muSet[c] = np.mean(EachFeatureSet, axis=0)
        covSet[c] = np.cov(EachFeatureSet.T)
        pcSet[c] = EachFeatureSet.shape[0]/FeatureExtractions.shape[0]
    return muSet, covSet, pcSet


def PGM_Predict(FeatureExtraction, muSet, covSet, pcSet): #Predict which class by PGM model.
    ySet, pos = {}, []
    for c in range(len(muSet)):
        ySet[c] = multivariate_normal.pdf(FeatureExtraction, mean=muSet[c], cov=covSet[c]) ##, allow_singular=True
    for c in range(len(muSet)):
        pos.append(ySet[c]*pcSet[c] / sum([ySet[i]*pcSet[i] for i in range(len(muSet))]))      
        predictClass, ProValue = max(enumerate(pos), key=operator.itemgetter(1))
    return predictClass, ProValue
        

def confusionMatrixPGM(FeatureExtractions, Labels, C, muSet, covSet, pcSet): #Calculating confusion matrix based on PGM model.
    if len(Labels.shape) > 1:
        Labels = Labels.reshape(Labels.shape[0])
    confusionMatrix = np.zeros((C,C))
    PredLabels = []
    for i in range(len(FeatureExtractions)):
        predictClass, ProValue = PGM_Predict(FeatureExtractions[i], muSet, covSet, pcSet)
        confusionMatrix[int(Labels[i]),predictClass] += 1
        PredLabels.append(predictClass)
    PredLabels = np.array(PredLabels)
    return confusionMatrix, PredLabels

def Scores(confusionMatrix): #Evaluate performance of confusion matrix by percision, recall, and accuracy.
    precisionSet, recallSet = [], []
    Ncor = 0
    for i in range(confusionMatrix.shape[0]):
        precisionSet.append(confusionMatrix[i,i]/np.sum(confusionMatrix[:,i]))
        recallSet.append(confusionMatrix[i,i]/np.sum(confusionMatrix[i,:]))
        Ncor += confusionMatrix[i,i]
    accuracySet = Ncor/np.sum(confusionMatrix)
    return accuracySet, precisionSet, recallSet


def CrossValidateForClassifier(EHDAveStorage, Labels, C, kFold=None):
    results = None
    tryTimes = 0
    while results is None and tryTimes < 20:
        try:
            Labels = Labels.reshape(Labels.shape[0],1) 
            dataShuSet = np.hstack((EHDAveStorage, Labels)) 
            np.random.shuffle(dataShuSet) #Shuffle the samples.
            trainSet = np.array_split(dataShuSet, kFold) #Divdie train data with K parts.
            parameters = {}
            overallTrainConfMat, overallValiConfMat = np.zeros((C,C)), np.zeros((C,C)) 
            subTrainRes, subValiRes, subTrainpParameters, overallTrainRes, overallValiRes = {},{},{},{},{}
            for k in range(kFold):
                #Data pre-process to divide k-1 train samples, and k validate samples.
                subTrainRes[k], subValiRes[k], subTrainpParameters[k] = {},{},{}
                subTrainRes[k]['Samples'], subValiRes[k]['Samples'] = {},{}
                trainSub = np.concatenate(np.delete(trainSet, k, 0)) #Store K-1 fold data together.
                valiSub = trainSet[k] #The K fold will be validating data.
                trainEHDAvgSub = trainSub[:,:-1]
                trainLabelsSub = trainSub[:,-1]
                valiEHDAvgSub = valiSub[:,:-1]
                valiLabelsSub = valiSub[:,-1]
                #Training and validating to output confusion matrix and scores.
                    #Training process.
                muSet, covSet, pcSet = PGM_Parameters(trainEHDAvgSub, trainLabelsSub)
                trainConfMat = confusionMatrixPGM(trainEHDAvgSub, trainLabelsSub, C, muSet, covSet, pcSet)
                trainAccuracy, trainPrecision, trainRecall = Scores(trainConfMat)
                overallTrainConfMat += trainConfMat
                    #Validating process
                valiConfMat = confusionMatrixPGM(valiEHDAvgSub, valiLabelsSub, C, muSet, covSet, pcSet)
                valiAccuracy, valiPrecision, valiRecall = Scores(valiConfMat)
                overallValiConfMat += valiConfMat
                #Save results including parameters, training, validating.
                subTrainpParameters[k]['mu'], subTrainpParameters[k]['cov'], subTrainpParameters[k]['pc'] = muSet, covSet, pcSet
                subTrainRes[k]['ConfMat'],subTrainRes[k]['Accuracy'],subTrainRes[k]['Precision'],subTrainRes[k]['Recall'], subTrainRes[k]['Samples']['Input'], subTrainRes[k]['Samples']['Labels'] \
                    = trainConfMat, trainAccuracy, trainPrecision, trainRecall, trainEHDAvgSub, trainLabelsSub
                subValiRes[k]['ConfMat'],subValiRes[k]['Accuracy'],subValiRes[k]['Precision'],subValiRes[k]['Recall'], subValiRes[k]['Samples']['Input'], subValiRes[k]['Samples']['Labels'] \
                    = valiConfMat, valiAccuracy, valiPrecision, valiRecall, valiEHDAvgSub, valiLabelsSub

            #Evaluate overall performance for training and validating, and saving.
            overallTrainAccuracy, overallTrainPrecision, overallTrainRecall = Scores(overallTrainConfMat)
            overallValiAccuracy, overallValiPrecision, overallValiRecall = Scores(overallValiConfMat)
            overallTrainRes['ConfMat'],overallTrainRes['Accuracy'],overallTrainRes['Precision'],overallTrainRes['Recall'] \
                  = overallTrainConfMat, overallTrainAccuracy, overallTrainPrecision, overallTrainRecall                                                                                                         
            overallValiRes['ConfMat'],overallValiRes['Accuracy'],overallValiRes['Precision'],overallValiRes['Recall'] \
                  = overallValiConfMat, overallValiAccuracy, overallValiPrecision, overallValiRecall

            #Save K fold training and validating reustls.
            CV_KfoldTrainValiRes = {}
            CV_KfoldTrainValiRes['train'], CV_KfoldTrainValiRes['vali'], CV_KfoldTrainValiRes['parameters'] \
                = subTrainRes, subValiRes, subTrainpParameters

            return CV_KfoldTrainValiRes, overallTrainRes, overallValiRes
            results = overallValiRes
        except:
            tryTimes += 1

def SplitDataToTrainTest (EHDAveStorage, Labels, TestSize):
    TrainLabels = np.copy(Labels)
    TrainLabels = TrainLabels.reshape(TrainLabels.shape[0],1)
    AlldataShu = np.hstack((EHDAveStorage, TrainLabels))
    np.random.shuffle(AlldataShu) #Shuffle the samples.
    d = round(AlldataShu.shape[0]*TestSize)
    TraindataShu = AlldataShu[:(AlldataShu.shape[0]-d)]
    TestdataShu = AlldataShu[(AlldataShu.shape[0]-d):]

    np.save("./TraindataShu%s" %(TestSize), TraindataShu)
    np.save("./TestdataShu%s" %(TestSize), TestdataShu)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Plot for heatmap. Reference form matplotlib.
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ticks = ["{:2.0f}".format(i) for i in cbar.get_ticks()]
    ticks[0] = 'None'
    cbar.ax.set_yticklabels(ticks) # set ticks of your format
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    Plot for heatmap. Reference form matplotlib.
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                continue
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
    
""" ======================  Variable Declaration ========================== """
##Declaration for to get best trained parameter file. 
C=5
BestKFold = 9 #Select best parameters in K fold.
BestFactors = (0, 1, 3) #Select best parameters by factors.
BestKfoldPath = "./CV_Results/F_(0, 1, 3)_K_9.npy"  #Read the best trained parameters for test.

""" =======================  Load Test Data ======================= """
Image = np.load('Images.npy')   #import test figures. The shape is (X, 200, 200, 3). 
Labels = np.load('Labels.npy')   #import test labels. The shape is (Y,).
Image = Image.astype(np.uint8)
Labels = Labels.astype(np.uint8)


""" ======================== Load Test the Model =========================== """

"""This is where you should load the testing data set. You shoud NOT re-train the model   """
Nimage = len(Image)
EHDStorage =[] 
EHDAveStorage =[] 
for i in range(0,Nimage):
    computer = EdgeHistogramComputer(4,4)
    img = Image[i]
    EHD = computer.compute(img)
    Average = np.array([sum(EHD[0::5]) / 16, sum(EHD[1::5]) / 16, sum(EHD[2::5]) / 16, sum(EHD[3::5]) / 16, sum(EHD[4::5]) / 16]).T
    EHDStor = np.append(EHD,Average)
    EHDStorage.append(EHDStor)
    EHDAveStorage.append(Average)
EHDAveStorage = np.array(EHDAveStorage)
EHDAveStorage = EHDAveStorage.reshape(EHDAveStorage.shape[0],EHDAveStorage.shape[2])
EHDAveTestFeatures = EHDAveStorage
EHDAveTestLabels =  Labels

print('-------------------------------------------------------------------------')
print('The model will process the raw images to Edge Histogram Descriptor (EHD) features.')
print('It will take some time for blind test set.')
print('-------------------------------------------------------------------------')

#To get best parameters.
BestKfoldFiles = np.load(BestKfoldPath,allow_pickle=True).tolist()
mu, cov, pc = BestKfoldFiles['parameters'][BestKFold-1]['mu'], BestKfoldFiles['parameters'][BestKFold-1]['cov'], BestKfoldFiles['parameters'][BestKFold-1]['pc']

#Test the model by  best K fold and factors parameters
EHDAveTestStorage = EHDAveTestFeatures[:, BestFactors] #Assign best factors for test.
#PredLabels is the output vector Y.
TestConfMat, PredLabels = confusionMatrixPGM(EHDAveTestStorage, EHDAveTestLabels, C, mu, cov, pc)
TestAccuracy, TestPrecision, TestRecall = Scores(TestConfMat)

print('-------------------------------------------------------------------------')
print('PredLabels is the output vector Y')
print('Using sklearn package to calculate the accuracy')
###Using sklearn package to calculate the accuracy.
from sklearn.metrics import accuracy_score
print('Accuracy: %s' %(accuracy_score(Labels, PredLabels)))
print('-------------------------------------------------------------------------')

print('-------------------------------------------------------------------------')
print('Test model accuracy:%s' %(TestAccuracy))
for i in range(len(TestPrecision)):
    print('Test model class %s precision:%s' %(i+1, TestPrecision[i]))
    print('Test model class %s recall:%s' %(i+1, TestRecall[i]))
print('-------------------------------------------------------------------------')

""" ========================  Plot Results ============================== """
""" This is where you should create the plots requested """
#Plot test confusion matrix results.
font = {'family' : 'Times New Roman',
        #'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

vegetables = ["C0","C1","C2","C3","C4"]

data = TestConfMat
harvest = np.array(data)

fig, ax = plt.subplots(figsize=(8,8))
im, cbar = heatmap(harvest, vegetables, vegetables, ax=ax,
                   cmap="Oranges", cbarlabel="Number")
texts = annotate_heatmap(im, valfmt="{x:2.0f}", size=20)
fig.tight_layout()

ax.set_title("Test results of confusion matrix")

plt.savefig("TestConfusionRes.png")
plt.show()

