#Coding a neural network
from operator import matmul
import matplotlib as plt
import numpy as np
from mnist import MNIST

#Trying to implement the gradient function

def activation(old_activation, bias, weight):
    # unstandardized_activation = np.zeros((1, len(weight[0])))
    print(old_activation.shape)
    print(bias.shape)
    print(weight.shape)
    #unst = np.matmul(weight, old_activation)
    #st = np.add(unst, bias)                        COMBINE THESE TWO TO SaVE MEMORY
    st = np.add(np.matmul(weight, old_activation), bias)
    print("\nPre sigmoud:\n{}".format(st))
    st_copy = st.copy()
    for i in range(len(st)):
        st_copy[i] = sigmoid(st[i])
    return st_copy
    # unst = np.vdot(weight, old_activation)
    # st = np.add(unst, bias)
    
    # unstandardized_activation = []
    # for i in range(len(bias)):
    #     unstandardized_activation.append(weight[i] * old_activation[i] + bias[i]) 
    
    # unstandardized_activation = np.array(unstandardized_activation)
    # standardized_activation = np.zeros(1, (len(unstandardized_activation)))
    # for i in range(len(unstandardized_activation)):    
    #     standardized_activation[i] = sigmoid(unstandardized_activation[i])
    # return standardized_activation

def sigmoid(activation):
    return 1/(1 + np.exp(-1 * activation))

def deriveSigmoid(activation):
    return 1/((1+np.exp(-1 * activation)) ** 2)



#The loss function should return the loss for a specific data point, which should be the sum of squared residuals
def calculateLoss(predicted, actual):
    sum = 0
    for i in range(predicted):
        sum += (actual[i]- predicted[i]) ** 2
    return sum 

def findGradient(oldActivation, predictedActivation, actual):
    gradient = np.zeros((len(oldActivation),1))
   
    # print(oldActivation.shape)
    # print(predictedActivation.shape)
    # print(actual.shape)
    # print(type(actual[0]))
    sum = 0
    for j in range(len(predictedActivation)):
        for i in range(len(oldActivation)):
            deriv = deriveSigmoid(float(actual[j]))
            print('stuff {} is the result of {} - {}'.format(predictedActivation[j] - actual[j],predictedActivation[j],actual[j]))
            #print('derivative{}'.format(deriv))
            #print('oldact{}'.format(oldActivation[i]))
            sum += (oldActivation[i] * 2 * deriv * (predictedActivation[j] - actual[j]))
            gradient[i] = sum
            sum = 0
    return gradient 

#
def updateWeights(weights, gradient, learningRate):
    negGradient = gradient
    for neg in negGradient:
        neg = neg * -1
    for i in range(len(weights)):
        weights[i] = weights[i] + negGradient[i] * learningRate
    return weights

mndata = MNIST('go')
trainImages, trainLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()
answer = int(trainLabels[0])
# print(answer)
actual = np.zeros((10,1))
actual[answer-1] = 1

# print(actual)
activation_values = np.array(trainImages[0]).reshape(784,1)




firstWeightMatrix = np.random.uniform(-1, 1,(28, 784))
firstBiasMatrix = np.random.uniform(-1, 1, (28, 1))



secondActivation = activation(activation_values, firstBiasMatrix, firstWeightMatrix)
secondWeightMatrix = np.random.uniform(-1, 1,(14, 28))
secondBiasMatrix = np.random.randint(-5, 5, size = (14, 1))



print("\n\n\nsecond activation\n{}".format(secondActivation))

#act = activation(activation_values, firstBiasMatrix[0], firstWeightMatrix[0])

thirdActivation = activation(secondActivation, secondBiasMatrix, secondWeightMatrix)
thirdWeightMatrix = np.random.uniform(-1, 1,(10, 14))
thirdBiasMatrix = np.random.randint(-5, 5, size = (10, 1))

predictedResults = activation(thirdActivation, thirdBiasMatrix, thirdWeightMatrix)
# print("\n\n\nPredicted results (first iteration)\n{}".format(predictedResults))

gradient = findGradient(thirdActivation, predictedResults, actual)
# print("------------")  
print(gradient)








        
