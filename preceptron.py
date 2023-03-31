#AUTHOR @ JACK VOLONTE
#DATE @ 9/25/2022
#DESCRIPTION @ This file will perform a step activation function for a preceptron w/data values provided from reading in histograms produced by accompaying program
#titled - 'histogramMaker.py'.

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time


# opening histogram files to read
file = open('AurHistogram.txt')
testingFile = open('validationAurs')


# getting random line from validation aurs data
def random_line_testing():
   line = random.choice(open('validationAurs').readlines())
   return line,1

# produces random line from testing histogram data file
def random_line():
   line = random.choice(open('AurHistogram.txt').readlines())
   return line,1

# returns summation of weights
def summationOfW(W,r):
   newW = 0.0
   for i in range(r-1):
      newW = -newW - (W[i] / W[r-1])
   return newW

# returns different summation of weights for each pixel grouping (r,g,b)
def summationOfWW(W,r):
   newW = 0.0
   for i in range(255):
      newW = -newW - (W[i] / W[i*2+1] / W[i*3+1])
   return newW

# returns summation of X1 data
def summationOfX1(X1,r):
   newX1 = 0.0
   for i in range(r-1):
      newX1 = -newX1 - (-X1[i] / X1[i + 1])
   return newX1

# returns dotproduct value between X1 and W data (histogram values and weights)
def dotX1W(X1,W,r):
   newDot = 0
   for i in range(r):
      newDot = newDot + np.dot(W[i], X1[i])
   return newDot

# returns the training error for that specific epoch
def trainingError(actual, predicted, epoch):
   if epoch != 0:
      newError = ((1/epoch) * (((actual - predicted)))**2) # **2 would be mse, this is rmse
   else:
      newError = (((actual - predicted)**2))
   return newError


# maths stuffs
epochs = input("Enter Epochs: ")
mu = input("Enter Learning Rate: ")  # learning rate
r = input("Enter Batch Size: ")
epochs = int(epochs)
mu = float(mu)
r = int(r)
countW = 0
accuracy = 0.0
error = 0.0
errors = np.zeros(epochs)
classificationAccuracy = np.zeros(epochs)


# preceptron step activation for loop
for y in range(epochs): #epochs

   # READING IN X1 DATA
   readingInX1,index = random_line() # reading in random lines from training file
   split_string = readingInX1.split("#", 1)
   substring = split_string[0]
   data = np.fromstring(substring, dtype=float, sep=' ')
   X1 = np.zeros(len(data) - 1)

   # READING IN X1
   readingIn = readingInX1
   split_string = readingIn.split("#", 1)
   substring = split_string[0]

   data = np.fromstring(substring, dtype=float, sep=' ')
   X1 = np.zeros(len(data))

   # assigning data to X1 variable
   count = 0
   for a in data:
      X1[count] = a
      count = count+1

   #plt.scatter(range(len(X1)),X1)


   #END OF READING IN X1
   if countW == 0:
      # randomizing weights (has to be done outside of epoch for ... reasons
      W = np.zeros(len(X1))
      for i in range(len(W)):
         W[i] = random.random()
      countW = countW+1


   # snagging L values from data
   L = np.full(len(X1), X1[0])
   X1[0] = 1 # setting first index of X1 to 0 for X_t later


   ones = np.ones(len(X1))
   #plt.scatter(ones, X1)
   # actual step activation loop
   for i in range(0,1):    # run this batch once
      activation = dotX1W(X1,W,r) # batch w/size
      #X1[0] = 1
      prediction = 1 if activation > 1 else -1
      print ("A: %f ,  P: %f , L: %f" %(activation, prediction,L[0]))
      if prediction == L[0]: #got it right
         accuracy += 1
      else:              #Error fix weights
         print("model adjust")
         error = prediction - L[i]
         print(error)
         W_t = W
         X1[0] = 1
         X_t = X1
         W_t = np.multiply(mu, np.multiply(error, X_t))
         W = np.subtract(W, W_t)
   errors[y] = trainingError(L[0], activation, y) # training error for size of epoch added after each iteration

   classificationAccuracy[y] = accuracy/epochs * 100


   plt.plot(np.arange(-.5,.5,1/len(X1)), (max(W)-min(W)) * np.arange(-.5,.5,1/len(X1)))
   #plt.scatter(np.arange(-.5,.5,1/len(X1)), np.arange(-.5,.5,1/len(X1))*X1)


# plt setup
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Preceptron of Aurora Borealis Training Data - %i Epochs' %epochs)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# end of plt setup

#plt.show()
# end of preceptron step activation for loop




# validation histograms loop
accuracy = 0.0

with open('validationAurs' ,'r') as f: # reading in file w/validation histograms
   countt = 0
   for line in f: # reading line by line from ^ - file w/validation histograms
      # READING IN X1 DATA
      readingInX1 = line
      split_string = readingInX1.split("#", 1)
      substring = split_string[0]
      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data) - 1)

      # READING IN X1
      readingIn = readingInX1
      split_string = readingIn.split("#", 1)
      substring = split_string[0]

      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data))

      # assigning data to X1 variable
      count = 0
      for a in data:
         X1[count] = a
         count = count + 1

      L = np.full(len(X1), X1[0]) # giving what the histogram should to the variable L
      X1[0] = 1  # setting first index of X1 to 0 for X_t later

      ones = np.ones(len(X1))

      # determining if the model is accurate for each histogram data sample
      r = 1
      for i in range(0,r):  # batch w/size
         #W = np.delete(W,0)
         #X1 = np.delete(X1, 0)
         X1[0] = 1
         activation = dotX1W(X1, W, len(X1))
         prediction = 1 if activation > 1 else -1
         print("A: %f ,  P: %f , L: %f" % (activation, prediction, L[0]))
         if prediction == L[0]: #got it right
            accuracy += 1
         else:              # error, model wrong
            print("model wrong")

   print('Total Accuracy: %i/150 validation images = %f' %(accuracy,accuracy/150*100))

plt.plot(np.arange(-.5, .5, 1 / len(W)), (max(W) - min(W)) * np.arange(-.5, .5, 1 / len(W))) # plotting W as a line
plt.show()

# end of validation histograms loop



# Start of plotting Training Error and Classification Accuracy vs Epoch

#plt.plot(np.arange(0,epochs,1),classificationAccuracy*(np.arange(0,epochs,1)))
plt.xlabel('Epochs')
plt.ylabel('Training Error (Blue) & Classification accuracy (Orange)')
plt.title('Training Error and Classificaton Accuracy for %i Epochs' %epochs)

plt.plot(np.arange(0,epochs,1), errors*np.arange(0,epochs,1), color="blue") # plotting training errors
plt.plot(np.arange(0,epochs,1), classificationAccuracy, color="orange") # classification accuracy

plt.show()
# End of plt setup

# End of plotting Training Error and Classification Accuracy vs Epoch




# Start of part 4 - checking if not unknowns are good

accuracy100 = 0
accuracy = 0
with open('AurUnknowns100.txt' ,'r') as f: # reading in file w/unknown histograms
   countt = 0
   for line in f: # reading line by line from ^ - file w/unknown histograms
      # READING IN X1 DATA
      readingInX1 = line
      split_string = readingInX1.split("#", 1)
      substring = split_string[0]
      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data) - 1)

      # READING IN X1
      readingIn = readingInX1
      split_string = readingIn.split("#", 1)
      substring = split_string[0]

      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data))

      # assigning data to X1 variable
      count = 0
      for a in data:
         X1[count] = a
         count = count + 1

      L = np.full(len(X1), X1[0]) # giving what the histogram should to the variable L
      X1[0] = 1  # setting first index of X1 to 0 for X_t later

      ones = np.ones(len(X1))

      # determining if the model is accurate for each histogram data sample
      r = 1
      for i in range(0,r):  # batch w/size
         #W = np.delete(W,0)
         #X1 = np.delete(X1, 0)
         X1[0] = 1
         activation = dotX1W(X1, W, len(X1))
         prediction = 1 if activation > 1 else -1
         #print("A: %f ,  P: %f , L: %f" % (activation, prediction, L[0]))
         if prediction == L[0]: #got it right
            accuracy += 1


   accuracy100 = accuracy


# End of checking if 100 unknowns accuracy



# Starting of checking if 300 unknowns accuracy

accuracy300 = 0
accuracy = 0
with open('AurUnknowns300.txt' ,'r') as f: # reading in file w/unknown histograms
   countt = 0
   for line in f: # reading line by line from ^ - file w/unknown histograms
      # READING IN X1 DATA
      readingInX1 = line
      split_string = readingInX1.split("#", 1)
      substring = split_string[0]
      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data) - 1)

      # READING IN X1
      readingIn = readingInX1
      split_string = readingIn.split("#", 1)
      substring = split_string[0]

      data = np.fromstring(substring, dtype=float, sep=' ')
      X1 = np.zeros(len(data))

      # assigning data to X1 variable
      count = 0
      for a in data:
         X1[count] = a
         count = count + 1

      L = np.full(len(X1), X1[0]) # giving what the histogram should to the variable L
      X1[0] = 1  # setting first index of X1 to 0 for X_t later

      ones = np.ones(len(X1))

      # determining if the model is accurate for each histogram data sample
      r = 1
      for i in range(0,r):  # batch w/size
         #W = np.delete(W,0)
         #X1 = np.delete(X1, 0)
         X1[0] = 1
         activation = dotX1W(X1, W, len(X1))
         prediction = 1 if activation > 1 else -1
         #print("A: %f ,  P: %f , L: %f" % (activation, prediction, L[0]))
         if prediction == L[0]: #got it right
            accuracy += 1

   accuracy300 = accuracy


# End of checking if 300 unkowns accuracy


print('Total Accuracy: %i/100 for first set of unknown images = %f' %(accuracy100,accuracy100/100*100))
print('Total Accuracy: %i/300 for second set of unknown images = %f' %(accuracy300,accuracy300/300*100))




