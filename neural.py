#import tensorflow_datasets as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits import mplot3d
import pandas as pd
import sys
import csv

np.set_printoptions(threshold=sys.maxsize)

class neuronLayer():

    def __init__(self, numOfNeurons, numOfInputsPerNeuron):

        #Each neuron layer is comprised of 2 matrices: one is the weights and the other is the biases
        #Initially, I just choose random numbers which will output random results
        self.synaptic_weights = (np.random.random((numOfInputsPerNeuron, numOfNeurons)) * 16) - 8
        self.biases = np.zeros((1, numOfNeurons))
        #self.biases = np.random.random((1, numOfNeurons)) - 0.5

class neuralNetwork():

    #This is the contrustor of the class. "self" is essentially the super keyword from java
    def __init__(self, layer1, layer2, layer3):

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

        self.layer1WeightAverage = np.zeros((784, 256))
        self.layer2WeightAverage = np.zeros((256, 256))
        self.layer3WeightAverage = np.zeros((256, 10))

        self.layer1BiasAverage = np.zeros((1, 256))
        self.layer2BiasAverage = np.zeros((1, 256))
        self.layer3BiasAverage = np.zeros((1, 10))

        fig = plt.figure()
        self.ax = plt.axes(projection = '3d')

    #input = np.linspace(-10, 10, 256)

    #I used the sigmoid function as my activation function
    #There are multiple sigmoids because some take in a matrix and return a sigmoided matrix and the other one just takes one value
    def sigmoid(self, x):

        output = (1 / (1 + np.exp(-(x))))

        return output

    def sigmoid2(self, x):

        output = np.zeros((1, 784))

        for i in range(len(x)):

            output[0][i] = 1/(1+np.exp(-(x[i])))

        return output

    def scale(self, x):

        for i in range(len(x)):

            x[i] = x[i] / 25.5

        return x

    #Sigmoid derivative needed for backpropagation algorithm
    def sigmoidPrime(self, x):

        #output = (1/(1 + np.exp(x))) * ( 1 - (1/(1 + np.exp(-(x)))))
        output = (np.exp(-(x))) * (1/((1 + np.exp(-(x))) ** 2))

        return output

    #This is the train method. I set a learning rate for both the bias and weights
    def train(self, input_pixels, desired_output, dataPoint):

        learningRateWeight = 0.3
        learningRateBias = 0.3

        input = []

        for i in range(0, 28):

            input = np.concatenate((input, input_pixels[i]))

        input = self.sigmoid2(self.scale(input))

        #I break up the input to separate the input and output from the input line from the text file
        #An example input looks like this: 21 90 63-0 0 0 1 0 0
        #The first 3 numbers are the rgb values and the numbers to the right of the rgb values is the desired output (this example means that the output neuron layer should say "0 red 0 orange 0 yellow 1 green 0 blue 0 white" which means the color is green)

        #This is where I send the inputs to the think method which computes the output through matrix multiplication
        #I record the output from each layer (obviously the output will be a number between 0 and 1 because it is sigmoided)
        outputFromLayer1, outputFromLayer2, outputFromLayer3 = self.think(input)
        trainingSetOutputs = [0] * 10
        trainingSetOutputs[desired_output] = 1


        #After getting the outputs, we can backpropagate (math is commented at the method itself)
        layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge = self.backprop(outputFromLayer3, outputFromLayer2, outputFromLayer1, trainingSetOutputs, input)

        '''
        self.layer1.synaptic_weights -= (layer1WeightNudges * learningRateWeight)
        self.layer2.synaptic_weights -= (layer2WeightNudges * learningRateWeight)
        self.layer3.synaptic_weights -= (layer3WeightNudges * learningRateWeight)

        self.layer1.biases -= (layer1BiasNudge * learningRateBias)
        self.layer2.biases -= (layer2BiasNudge * learningRateBias)
        self.layer3.biases -= (layer3BiasNudge * learningRateBias)

        '''
        self.layer1WeightAverage += (layer1WeightNudges)
        self.layer2WeightAverage += (layer2WeightNudges)
        self.layer3WeightAverage += (layer3WeightNudges)

        self.layer1BiasAverage += (layer1BiasNudge)
        self.layer2BiasAverage += (layer2BiasNudge)
        self.layer3BiasAverage += (layer3BiasNudge)

        if dataPoint % 10 == 0:

            self.layer1.synaptic_weights -= (((self.layer1WeightAverage * learningRateWeight) / 10))
            self.layer2.synaptic_weights -= (((self.layer2WeightAverage * learningRateWeight) / 10))
            self.layer3.synaptic_weights -= (((self.layer3WeightAverage * learningRateWeight) / 10))

            self.layer1.biases -= ((self.layer1BiasAverage / 10) * learningRateBias)
            self.layer2.biases -= ((self.layer2BiasAverage / 10) * learningRateBias)
            self.layer3.biases -= ((self.layer3BiasAverage / 10) * learningRateBias)

            self.layer1WeightAverage = np.zeros((784, 256))
            self.layer2WeightAverage = np.zeros((256, 256))
            self.layer3WeightAverage = np.zeros((256, 10))

            self.layer1BiasAverage = np.zeros((1, 256))
            self.layer2BiasAverage = np.zeros((1, 256))
            self.layer3BiasAverage = np.zeros((1, 10))


    #And now for the meat of the network
    def backprop(self, outputFromLayer3, outputFromLayer2, outputFromLayer1, desiredOutput, trainingSetInputs):

        #I create empty 1 X 6 matrices to record the nudges after I compute them
        layer3WeightNudges = np.zeros((256, 10))
        layer2WeightNudges = np.zeros((256, 256))
        layer1WeightNudges = np.zeros((784, 256))
        layer2ActivationNudge = np.zeros((1, 256))
        layer1ActivationNudge = np.zeros((1, 256))
        layer0ActivationNudge = np.zeros((1, 784))

        layer1BiasNudge = np.zeros((1, 256))
        layer2BiasNudge = np.zeros((1, 256))
        layer3BiasNudge = np.zeros((1, 10))

        costDer = 2 * (outputFromLayer3 - desiredOutput)
        layer3BiasNudge = self.sigmoidPrime(outputFromLayer3) * costDer
        activation = layer3BiasNudge
        layer3WeightNudges = np.dot(np.transpose(outputFromLayer2), activation)
        activation = np.dot(self.layer3.synaptic_weights, np.transpose(activation))
        layer2BiasNudge = np.transpose(activation) * self.sigmoidPrime(outputFromLayer2)
        activation = layer2BiasNudge
        layer2WeightNudges = np.dot(np.transpose(outputFromLayer1), activation)
        activation = np.dot(self.layer2.synaptic_weights, np.transpose(activation))
        layer1BiasNudge = np.transpose(activation) * self.sigmoidPrime(outputFromLayer1)
        activation = layer1BiasNudge
        layer1WeightNudges = np.dot(np.transpose(trainingSetInputs), activation)

        return layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge

    #This thinking part is just multiplying the matrices to get the output layer. I didn't find a matrix mutliplication method that suits my needs so I just wrote it but there may be one out there
    def think(self, inputs):

        outputFromLayer1 = self.sigmoid(np.dot(inputs, self.layer1.synaptic_weights) + self.layer1.biases)
        outputFromLayer2 = self.sigmoid(np.dot(outputFromLayer1, self.layer2.synaptic_weights) + self.layer2.biases)
        outputFromLayer3 = self.sigmoid(np.dot(outputFromLayer2, self.layer3.synaptic_weights) + self.layer3.biases)

        print(outputFromLayer3)

        return outputFromLayer1, outputFromLayer2, outputFromLayer3

    def WriteToCSV(self, fileName, outputMatrix, rows, cols):

        with open(fileName, mode = 'w') as output_file:
            output_file = csv.writer(output_file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)

            for i in range(rows):

                for k in range(cols):

                    output_file.writerow([str(outputMatrix[i][k])])

    #The test method is the same as the train method minus all the math. I just get the output
    def test(self, testData):

        input = []

        for i in range(0, 28):

            input = np.concatenate((input, testData[i]))

        input = self.sigmoid2(self.scale(input))

        #trainingSetInputs = self.sigmoid2(trainingSetInputs)
        outputsFromLayer1, outputsFromLayer2, outputsFromLayer3 = self.think(input)

        self.WriteToCSV("layer1OutputWeights.csv", self.layer1.synaptic_weights, 784, 256)
        self.WriteToCSV("layer2OutputWeights.csv", self.layer2.synaptic_weights, 256, 256)
        self.WriteToCSV("layer3OutputWeights.csv", self.layer3.synaptic_weights, 256, 10)

        self.WriteToCSV("layer1OutputBias.csv", self.layer1.biases, 1, 256)
        self.WriteToCSV("layer2OutputBias.csv", self.layer2.biases, 1, 256)
        self.WriteToCSV("layer3OutputBias.csv", self.layer3.biases, 1, 10)

        '''
        fileWeight = open("outputWeights.txt", "w")
        fileWeight.write("layer1: ")
        fileWeight.write(str(self.layer1.synaptic_weights))
        fileWeight.write("layer2: ")
        fileWeight.write(str(self.layer2.synaptic_weights))
        fileWeight.write("layer3: ")
        fileWeight.write(str(self.layer3.synaptic_weights))
        fileWeight.close()

        fileBias = open("outputBiases.txt", "w")
        fileBias.write("layer1: ")
        fileBias.write(str(self.layer1.biases))
        fileBias.write("layer2: ")
        fileBias.write(str(self.layer2.biases))
        fileBias.write("layer3: ")
        fileBias.write(str(self.layer3.biases))
        fileBias.close()

        for i in range(0, 10):

            print(outputsFromLayer3[0][i], " ")
        '''

#This is the "public static void main" method for python
if __name__ == "__main__":

    np.random.seed(1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    layer1 = neuronLayer(256, 784)
    layer2 = neuronLayer(256, 256)
    layer3 = neuronLayer(10, 256)

    neuralNet = neuralNetwork(layer1, layer2, layer3)

    for i in range(5):

        for t in range(60000):

            neuralNet.train(x_train[t], y_train[t], t + 1)
            print("Done training with the ", t + 1, "th data. On to the ", t + 2,"th data")

    neuralNet.test(x_test[12])
    print("Finished the training and testing of the neural network")
    print("The desired output of the test is: ", y_test[12])
