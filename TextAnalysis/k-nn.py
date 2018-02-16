import csv
from random import shuffle
import math
from collections import Counter
import numpy as np



def loadInputFiles():
    global trainingData,testData,stopWords
    datafile = open('Combined_News_DJIA.csv', 'r',newline="\n")
    dataReader = csv.reader(datafile,delimiter=',')
    data = list(dataReader)
    data=data[1:]
    shuffle(data)
    datafile.close()

    dataSize=len(data)
    trainingDataSize= math.ceil((80*dataSize)/100)
    trainingData=data[0:trainingDataSize]
    testData=data[trainingDataSize:]
    return trainingData,testData


def getBigrams(news):
    words = news.split()
    bigrams = []
    numWords = len(words)
    for i in range(numWords - 1):
        bigrams.append((words[i], words[i + 1]))
    return bigrams


def getTrigrams(news):
    words = news.split()
    trigrams = []
    numWords = len(words)
    for i in range(numWords - 2):
        trigrams.append((words[i], words[i + 1], words[i + 2]))
    return trigrams


def getTermFrequency(bigrams):
    N= len(trainingData)
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    #counter = Counter(freq_dict)
    return freq_dict


def getIDF(freq):
    N=len(trainingData)
    for x in freq:
        freq[x]=math.log10(N/(freq.get(x)+1))

    return freq


def extractNGrams(trainingData):
    allBigrams=[]
    allTrigrams=[]
    for i in range(len(trainingData)):
        for j in range(2,len(trainingData[i])):
            bigrams=getBigrams(trainingData[i][j])
            trigrams=getTrigrams(trainingData[i][j])
            allBigrams.extend(bigrams)
            allTrigrams.extend(trigrams)
    return allBigrams,allTrigrams


def generateTFVectors(biTriGram):
    vocab=set(biTriGram)
    vocab=list(vocab)
    vocab=np.array(vocab)
    tfVector=np.zeros(len(trainingData),dtype=object)
    #tfVector2=np.zeros((len(trainingData),len(vocab)))
    for i in range(len(trainingData)):
        newsBiTrigram=[]
        for j in range(2,len(trainingData[i])):
            newsBiTrigram.extend(getBigrams(trainingData[i][j]))
            newsBiTrigram.extend(getTrigrams(trainingData[i][j]))
        #newsBigram=np.array(newsBigram)
        freq_dict = {}
        for biTrigram in newsBiTrigram:
            #indx=np.where((vocab==bigram).all(axis=1))
            if freq_dict.get(biTrigram):
                freq_dict[biTrigram] += 1
            else:
                freq_dict[biTrigram] = 1
        tfVector[i]=freq_dict

    print('yni')
    return tfVector


def normalizeTFVector(tfVector):
    for i in range(1,len(tfVector)):
            count = 0
            for x in tfVector[i]:
                count+=tfVector[i].get(x)
            for x in tfVector[i]:
                tfVector[i][x]=tfVector[i].get(x)/count
    return tfVector


def normalizeTestTFVector(tfVector):
            count = 0
            for x in tfVector:
                count+=tfVector.get(x)
            for x in tfVector:
                tfVector[x]=tfVector.get(x)/count
            return tfVector


def getTFVector(biTrigram):
    freq_dict = {}
    for gram in biTrigram:
        # indx=np.where((vocab==bigram).all(axis=1))
        if freq_dict.get(gram):
            freq_dict[gram] += 1
        else:
            freq_dict[gram] = 1
    return freq_dict


def calculateCosineSimilarity(trainTFVector,trainIDFVector,tstData):
    similarityResults=[]
    for testDoc in tstData:
        simRes=[]
        biTrigram=[]
        for i in range(2,len(testDoc)):
            bigram=getBigrams(testDoc[i])
            trigram=getTrigrams(testDoc[i])
            biTrigram.extend(bigram)
            biTrigram.extend(trigram)
        testDocTFVector=getTFVector(biTrigram)
        normalTestDocTFVector=normalizeTestTFVector(testDocTFVector)
        testIdfVector=getIDF(testDocTFVector)
        testTFSet=set(normalTestDocTFVector)
        for i in  range(len(trainTFVector)):
            trainTFSet=set(trainTFVector[i])
            sumOfMul=0
            sumOfDenominatorTrain=0
            sumOfDenominatorTest=0
            for key in testTFSet.intersection(trainTFSet):
                trainTFIDF=trainTFVector[i][key] * trainIDFVector[key]
                sumOfDenominatorTrain+=math.pow(trainTFIDF,2)
                testTFIDF=normalTestDocTFVector[key] * testIdfVector[key]
                sumOfDenominatorTest += math.pow(testTFIDF, 2)
                sumOfMul+=trainTFIDF * testTFIDF
            if sumOfDenominatorTrain==0 or sumOfDenominatorTest==0:
                cosTheta=0
            else:
                cosTheta= sumOfMul/(math.sqrt(sumOfDenominatorTrain) * math.sqrt(sumOfDenominatorTest))
            simRes.append(cosTheta)
        similarityResults.append(simRes)

    return similarityResults


def performKnn(similarityResults,K,trnData,tstData):
    correct=0
    for i,testSampleSim in enumerate(similarityResults):
        testSampleSim=np.array(testSampleSim)
        prediction=-1
        #mostSimilars=np.argsort(-testSampleSim)
        kNN=testSampleSim.argsort()[::-1][:K]
        classOne=0
        classZero=0
        for j in kNN:
            if trnData[j][1] ==1:
                classOne+=1
            else: classZero+=1
        if classOne>classZero:
            prediction=1
        else:
            prediction=0
        if int(tstData[i][1])==prediction:
            correct+=1

    return (correct/len(tstData) )*100


def generateTFIDFVectors(normalTFVector,idfVector):
    tfidfVec=[]
    for tf in normalTFVector:
        tfidfDoc=[]
        for key in tf:
            tfidf=tf[key] * idfVector[key]
            tfidfDoc.append({key:tfidf})
        tfidfVec.append(tfidfDoc)
    return tfidfVec

trainingData, testData=loadInputFiles()
allBigrams,allTrigrams=extractNGrams(trainingData)
frequencies=getTermFrequency(allBigrams+allTrigrams)
idfVector=getIDF(frequencies)
tfVector=generateTFVectors(allBigrams+allTrigrams)
normalTFVector=normalizeTFVector(tfVector)
tfidfVector=generateTFIDFVectors(normalTFVector,idfVector)
tfidfVector=np.array(tfidfVector)
np.savetxt("tf.csv", normalTFVector,fmt="%s", delimiter=",")
np.savetxt("tf-idf.csv", tfidfVector,fmt="%s", delimiter=",")

K=40

similarityResults=calculateCosineSimilarity(normalTFVector,idfVector,testData)  # for test and train change 3rd param

acc =performKnn(similarityResults,K,trainingData,testData) # for test and train change 3rd & 4th params
print('Test acc with K : ', K)
print(acc)
'''
similarityResults=calculateCosineSimilarity(normalTFVector,idfVector,trainingData)

acc =performKnn(similarityResults,K,trainingData,trainingData) # for test and train change 3rd & 4th params
print('Train acc with K : ', K)
print(acc)
'''