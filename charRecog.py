#This code currently only recognises 3 characters
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave
import math
import re
import glob
import cv2
import scipy
import cPickle as pickle

#Discriminant function for classification
def discriminant(test,mean,invCov,logdet): 
	
	tr = (test-mean).T 
	temp = np.dot(invCov,(test-mean)) 
	return -0.5*np.dot(tr,temp) - 0.5*logdet

#Function to sort files
digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


#1(a)
#Taking collapsed image array as feature vector
I1 = np.zeros((49*49,600),dtype=np.float64)
files = glob.glob("TrainCharacters/1/*.jpg")
files.sort(key = tokenize)
for i in range(len(files)):
        i1 = cv2.imread(files[i])
	i1 = i1[14:112, 14:112]
	i1 = cv2.resize(i1,(0,0),fx=0.5, fy=0.5)
	temp = i1[:,:,0]
	I1[:,i] = temp.ravel()	

#In this case, we calculate covariance matrices for each character class
lam=100	#Regularisation parameter, set to optimum value, change to get different results
I = np.eye(2401)
regTerm = lam*I
meanI1 = np.mean(I1, axis=1)
covMatrix1 = np.cov(I1)
sigma1 = covMatrix1 + regTerm
sigmaInv1 = np.linalg.inv(sigma1)
detSigmaReg1 = np.linalg.slogdet(sigma1)[1]

I2 = np.zeros((49*49,600),dtype=np.float64)


files = glob.glob("TrainCharacters/2/*.jpg")
files.sort(key = tokenize)
for i in range(len(files)):
        i2 = cv2.imread(files[i])
        i2 = i2[14:112, 14:112]
        i2 = cv2.resize(i2,(0,0),fx=0.5, fy=0.5)
        temp = i2[:,:,0]
        I2[:,i] = temp.ravel()
        
covMatrix2 = np.cov(I2)
meanI2 = np.mean(I2, axis=1)
covMatrix2 = np.cov(I2)
sigma2 = covMatrix2 + regTerm
sigmaInv2 = np.linalg.inv(sigma2)
detSigmaReg2 = np.linalg.slogdet(sigma2)[1]


I3 = np.zeros((49*49,600),dtype=np.float64)

files = glob.glob("TrainCharacters/3/*.jpg")
files.sort(key = tokenize)
for i in range(len(files)):
        i3 = cv2.imread(files[i])
        i3 = i3[14:112, 14:112]
        i3 = cv2.resize(i3,(0,0),fx=0.5, fy=0.5)
        temp = i3[:,:,0]
        I3[:,i] = temp.ravel()

meanI3 = np.mean(I3, axis=1)
covMatrix3 = np.cov(I3)
sigma3 = covMatrix3 + regTerm
sigmaInv3 = np.linalg.inv(sigma3)
detSigmaReg3 = np.linalg.slogdet(sigma3)[1]

testFiles = glob.glob("TestCharacters/*/*.jpg")
testFiles.sort(key = tokenize)

test = np.zeros((49*49,600),dtype=np.float64)
for i in range(len(testFiles)):
        t1 = cv2.imread(testFiles[i])
        t1 = t1[14:112, 14:112]
        t1 = cv2.resize(t1,(0,0),fx=0.5, fy=0.5)
        temp = t1[:,:,0]
        test[:,i] = temp.ravel()

predict = np.zeros((300,2))
accuracy = np.zeros((1,3))
for i in xrange(len(testFiles)):
	g1 = discriminant(test[:,i], meanI1, sigmaInv1, detSigmaReg1)
	g2 = discriminant(test[:,i], meanI2, sigmaInv2, detSigmaReg2)
	g3 = discriminant(test[:,i], meanI3, sigmaInv3, detSigmaReg3)
        if g1>g2 and g1>g3:
        	predict[i,1] = 1
        if g2>g1 and g2>g3:
                predict[i,1] = 2
        if g3>g1 and g3>g1:
                predict[i,1] = 3
for i in xrange(0,100):
        predict[i,0] = 1
        if predict[i,0]==predict[i,1]:
        	accuracy[0,0]+=1
for i in xrange(100,200):
        predict[i,0] = 2
        if predict[i,0]==predict[i,1]:
                accuracy[0,1]+=1
for i in xrange(200,300):
        predict[i,0] = 3
        if predict[i,0]==predict[i,1]:
                accuracy[0,2]+=1

print accuracy


#In this case , we take a common covariance for all character classes
I = np.zeros((49*49,600),dtype=np.float64)

files = glob.glob("TrainCharacters/*/*.jpg")
files.sort(key = tokenize)
for i in range(len(files)):
        it = cv2.imread(files[i])
        it = it[14:112, 14:112]
        it = cv2.resize(it,(0,0),fx=0.5, fy=0.5)
        temp = it[:,:,0]
        I[:,i] = temp.ravel()

meanI = np.mean(I,axis=1)
covMatrix = np.cov(I)
sigma = covMatrix1 + regTerm
sigmaInv = np.linalg.inv(sigma)
detSigmaReg = np.linalg.slogdet(sigma)[1]
accuracy =  np.zeros((1,3))
for i in xrange(len(testFiles)):
        g1 = discriminant(test[:,i], meanI1, sigmaInv, detSigmaReg)
        g2 = discriminant(test[:,i], meanI2, sigmaInv, detSigmaReg)
        g3 = discriminant(test[:,i], meanI3, sigmaInv, detSigmaReg)
        if g1>g2 and g1>g3:
                predict[i,1] = 1
        if g2>g1 and g2>g3:
                predict[i,1] = 2
        if g3>g1 and g3>g1:
                predict[i,1] = 3
for i in xrange(0,100):
        predict[i,0] = 1
        if predict[i,0]==predict[i,1]:
                accuracy[0,0]+=1
for i in xrange(100,200):
        predict[i,0] = 2
        if predict[i,0]==predict[i,1]:
                accuracy[0,1]+=1
for i in xrange(200,300):
        predict[i,0] = 3
        if predict[i,0]==predict[i,1]:
                accuracy[0,2]+=1

print accuracy


#In this case, samples are modelled using a diagonal covariance matrix for each class

sigmaDia1 = np.multiply(sigma1,(np.eye(2401)))
sigmaDia2 = np.multiply(sigma2,(np.eye(2401)))
sigmaDia3 = np.multiply(sigma3,(np.eye(2401)))


sigmaDiaInv1 = np.linalg.inv(sigmaDia1)
sigmaDiaInv2 = np.linalg.inv(sigmaDia2)
sigmaDiaInv3 = np.linalg.inv(sigmaDia3)

sigmaDiaDet1 = np.linalg.slogdet(sigmaDia1)[1]
sigmaDiaDet2 = np.linalg.slogdet(sigmaDia2)[1]
sigmaDiaDet3 = np.linalg.slogdet(sigmaDia3)[1]

accuracy =  np.zeros((1,3))
for i in xrange(len(testFiles)):
        g1 = discriminant(test[:,i], meanI1, sigmaDiaInv1, sigmaDiaDet1)
        g2 = discriminant(test[:,i], meanI2, sigmaDiaInv2, sigmaDiaDet2)
        g3 = discriminant(test[:,i], meanI3, sigmaDiaInv3, sigmaDiaDet3)
        if g1>g2 and g1>g3:
                predict[i,1] = 1
        if g2>g1 and g2>g3:
                predict[i,1] = 2
        if g3>g1 and g3>g1:
                predict[i,1] = 3
for i in xrange(0,100):
        predict[i,0] = 1
        if predict[i,0]==predict[i,1]:
                accuracy[0,0]+=1
for i in xrange(100,200):
        predict[i,0] = 2
        if predict[i,0]==predict[i,1]:
                accuracy[0,1]+=1
for i in xrange(200,300):
        predict[i,0] = 3
        if predict[i,0]==predict[i,1]:
                accuracy[0,2]+=1

print accuracy

#In this case, samples are pooled to get a common diagonal covariance matrix

sigmaDia = np.multiply(sigma,(np.eye(2401)))
sigmaDiaInv = np.linalg.inv(sigmaDia)
sigmaDiaDet = np.linalg.slogdet(sigmaDia)[1]
accuracy =  np.zeros((1,3))
for i in xrange(len(testFiles)):
        g1 = discriminant(test[:,i], meanI1, sigmaDiaInv, sigmaDiaDet)
        g2 = discriminant(test[:,i], meanI2, sigmaDiaInv, sigmaDiaDet)
        g3 = discriminant(test[:,i], meanI3, sigmaDiaInv, sigmaDiaDet)
        if g1>g2 and g1>g3:
                predict[i,1] = 1
        if g2>g1 and g2>g3:
                predict[i,1] = 2
        if g3>g1 and g3>g1:
                predict[i,1] = 3
for i in xrange(0,100):
        predict[i,0] = 1
        if predict[i,0]==predict[i,1]:
                accuracy[0,0]+=1
for i in xrange(100,200):
        predict[i,0] = 2
        if predict[i,0]==predict[i,1]:
                accuracy[0,1]+=1
for i in xrange(200,300):
        predict[i,0] = 3
        if predict[i,0]==predict[i,1]:
                accuracy[0,2]+=1

print accuracy


#The covariance of each class is forced to be spherical

varSph1 = np.trace(sigmaDia1)/2401*np.eye(2401)
varSph2 = np.trace(sigmaDia2)/2401*np.eye(2401)
varSph3 = np.trace(sigmaDia3)/2401*np.eye(2401)

sigmaSphInv1 = np.linalg.inv(varSph1)
sigmaSphInv2 = np.linalg.inv(varSph2)
sigmaSphInv3 = np.linalg.inv(varSph3)

detSph1 = 2*2401*np.log(np.trace(sigmaDia1)/2401)
detSph2 = 2*2401*np.log(np.trace(sigmaDia2)/2401)
detSph3 = 2*2401*np.log(np.trace(sigmaDia3)/2401)
accuracy =  np.zeros((1,3))
for i in xrange(len(testFiles)):
        g1 = discriminant(test[:,i], meanI1, sigmaSphInv1, detSph1)
        g2 = discriminant(test[:,i], meanI2, sigmaSphInv2, detSph2)
        g3 = discriminant(test[:,i], meanI3, sigmaSphInv3, detSph3)
        if g1>g2 and g1>g3:
                predict[i,1] = 1
        if g2>g1 and g2>g3:
                predict[i,1] = 2
        if g3>g1 and g3>g1:
                predict[i,1] = 3
for i in xrange(0,100):
        predict[i,0] = 1
        if predict[i,0]==predict[i,1]:
                accuracy[0,0]+=1

for i in xrange(100,200):
        predict[i,0] = 2
        if predict[i,0]==predict[i,1]:
                accuracy[0,1]+=1

for i in xrange(200,300):
        predict[i,0] = 3
        if predict[i,0]==predict[i,1]:
                accuracy[0,2]+=1

print accuracy

