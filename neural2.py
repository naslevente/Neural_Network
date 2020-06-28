#import tensorflow_datasets as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import time

layer1WeightAverage = 0;
layer2WeightAverage = 0;
layer3WeightAverage = 0;

layer1BiasAverage = 0;
layer2BiasAverage = 0;
layer3BiasAverage = 0;

class neuronLayer():

    def __init__(self, numOfNeurons, numOfInputsPerNeuron):

        #Each neuron layer is comprised of 2 matrices: one is the weights and the other is the biases
        #Initially, I just choose random numbers which will output random results
        self.synaptic_weights = np.random.random((numOfInputsPerNeuron, numOfNeurons)) * 0.01
        self.biases = np.zeros((1, numOfNeurons))
        #self.biases = np.random.random((1, numOfNeurons)) - 0.5

class neuralNetwork():

    #This is the contrustor of the class. "self" is essentially the super keyword from java
    def __init__(self, layer1, layer2, layer3):

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    #input = np.linspace(-10, 10, 100)

    #I used the sigmoid function as my activation function
    #There are multiple sigmoids because some take in a matrix and return a sigmoided matrix and the other one just takes one value
    def sigmoid(self, x):

        if len(x[0]) == 16: output = np.zeros((1, 16))
        else: output = np.zeros((1, 10))

        for i in range(x.size):

            output[0][i] = 1/(1+np.exp(-(x[0][i])))

        return output

    def sigmoid2(self, x):

        output = np.zeros((1, 784))

        for i in range(len(x)):

            output[0][i] = 1/(1+np.exp(-(x[i])))

        return output

    def scale(self, x):

        for i in range(len(x)):

            x[i] = x[i] / 255

        return x

    #Sigmoid derivative needed for backpropagation algorithm
    def sigmoidPrime(self, x):

        output = (1/(1 + np.exp(x))) * ( 1 - (1/(1 + np.exp(-(x)))))

        return output

    #This is the train method. I set a learning rate for both the bias and weights
    def train(self, input_pixels, desired_output, dataPoint):

        print("WEIGHTS")
        print(self.layer1.synaptic_weights)
        print(self.layer2.synaptic_weights)
        print(self.layer3.synaptic_weights)
        print("DONE")

        learningRateWeight = 0.3
        learningRateBias = 0.3

        input = []

        for i in range(0, 28):

            input = np.concatenate((input, input_pixels[i]))

        input = self.scale(input)

        #I break up the input to separate the input and output from the input line from the text file
        #An example input looks like this: 21 90 63-0 0 0 1 0 0
        #The first 3 numbers are the rgb values and the numbers to the right of the rgb values is the desired output (this example means that the output neuron layer should say "0 red 0 orange 0 yellow 1 green 0 blue 0 white" which means the color is green)

        #This is where I send the inputs to the think method which computes the output through matrix multiplication
        #I record the output from each layer (obviously the output will be a number between 0 and 1 because it is sigmoided)
        outputFromLayer1, outputFromLayer2, outputFromLayer3 = self.think(input)
        trainingSetOutputs = [0] * 10
        trainingSetOutputs[desired_output] = 1

        print("OUTPUTS:")
        print(outputFromLayer1)
        print(outputFromLayer2)
        print(outputFromLayer3)
        print("DONE")

        #After getting the outputs, we can backpropagate (math is commented at the method itself)
        layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge = self.backprop(outputFromLayer3, outputFromLayer2, outputFromLayer1, trainingSetOutputs, input)

        print("WEIGHT NUDGES")
        print(layer3WeightNudges)
        print(layer2WeightNudges)
        print(layer1WeightNudges)
        print("DONE")

        print("BIAS NUDGES")
        print(layer3BiasNudge)
        print(layer2BiasNudge)
        print(layer1BiasNudge)
        print("DONE")

        #The idea is to average the nudges through all the training examples so that the network doesn't learn to recognize one specific example (Rather than applying many partially incorrect nudges, it applies some precise nudges)
        global layer1WeightAverage
        global layer2WeightAverage
        global layer3WeightAverage

        global layer1BiasAverage
        global layer2BiasAverage
        global layer3BiasAverage

        layer1WeightAverage += (layer1WeightNudges)
        layer2WeightAverage += (layer2WeightNudges)
        layer3WeightAverage += (layer3WeightNudges)

        layer1BiasAverage += (layer1BiasNudge)
        layer2BiasAverage += (layer2BiasNudge)
        layer3BiasAverage += (layer3BiasNudge)

        if dataPoint % 100 == 0:

            self.layer1.synaptic_weights -= ((layer1WeightAverage / 100) * learningRateWeight)
            self.layer2.synaptic_weights -= ((layer2WeightAverage / 100) * learningRateWeight)
            self.layer3.synaptic_weights -= ((layer3WeightAverage / 100) * learningRateWeight)

            self.layer1.biases -= ((layer1BiasAverage / 100) * learningRateBias)
            self.layer2.biases -= ((layer2BiasAverage / 100) * learningRateBias)
            self.layer3.biases -= ((layer3BiasAverage / 100) * learningRateBias)

            layer1WeightAverage = 0
            layer2WeightAverage = 0
            layer3WeightAverage = 0

            layer1BiasAverage = 0
            layer2BiasAverage = 0
            layer3BiasAverage = 0

    #And now for the meat of the network
    def backprop(self, outputFromLayer3, outputFromLayer2, outputFromLayer1, desiredOutput, trainingSetInputs):

        print("CHECK: ", len(trainingSetInputs))
        #I create empty 1 X 6 matrices to record the nudges after I compute them
        layer3WeightNudges = np.zeros((16, 10))
        layer2WeightNudges = np.zeros((16, 16))
        layer1WeightNudges = np.zeros((784, 16))
        #layer2ActivationNudge = np.zeros((1, 16))
        #layer1ActivationNudge = np.zeros((1, 16))
        #layer0ActivationNudge = np.zeros((1, 784))

        layer1BiasNudge = np.zeros((1, 16))
        layer2BiasNudge = np.zeros((1, 16))
        layer3BiasNudge = np.zeros((1, 10))

        costDer = 2 * (outputFromLayer3 - desiredOutput)
        layer3BiasNudge = self.sigmoidPrime(outputFromLayer3) * costDer
        activation = np.transpose(layer3BiasNudge)
        layer3WeightNudges = np.dot(activation, outputFromLayer2)
        activation = np.dot(self.layer3.synaptic_weights, activation)
        layer2BiasNudge = np.transpose(activation)
        layer2WeightNudges = np.dot(np.transpose(layer2BiasNudge * self.sigmoidPrime(outputFromLayer2)), outputFromLayer1)
        activation = np.dot(self.layer2.synaptic_weights, activation)
        layer1BiasNudge = activation
        layer1WeightNudges = np.dot(np.transpose(layer1BiasNudge * self.sigmoidPrime(outputFromLayer1)), trainingSetInputs)

        return layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge

    #This thinking part is just multiplying the matrices to get the output layer. I didn't find a matrix mutliplication method that suits my needs so I just wrote it but there may be one out there
    def think(self, inputs):

        print("INPUTS")
        print(inputs)
        print("DONE")

        outputFromLayer1 = self.sigmoid(np.dot(self.sigmoid2(inputs), self.layer1.synaptic_weights) - self.layer1.biases)
        outputFromLayer2 = self.sigmoid(np.dot(outputFromLayer1, self.layer2.synaptic_weights) - self.layer2.biases)
        outputFromLayer3 = self.sigmoid(np.dot(outputFromLayer2, self.layer3.synaptic_weights) - self.layer3.biases)

        print(self.layer3.synaptic_weights.shape)
        print(outputFromLayer2.shape)

        return outputFromLayer1, outputFromLayer2, outputFromLayer3


    #The test method is the same as the train method minus all the math. I just get the output
    def test(self, testData):

        input = []

        for i in range(0, 28):

            input = np.concatenate((input, testData[i]))

        #trainingSetInputs = self.sigmoid2(trainingSetInputs)
        outputsFromLayer1, outputsFromLayer2, outputsFromLayer3 = self.think(input)

        print(self.layer1.synaptic_weights)
        print(self.layer2.synaptic_weights)
        print(self.layer3.synaptic_weights)

        for i in range(0, 10):

            print(outputsFromLayer3[0][i], " ")

#This is the "public static void main" method for python
if __name__ == "__main__":

    np.random.seed(1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    layer1 = neuronLayer(16, 784)
    layer2 = neuronLayer(16, 16)
    layer3 = neuronLayer(10, 16)

    neuralNet = neuralNetwork(layer1, layer2, layer3)

    for i in range(1):

        neuralNet.train(x_train[i], y_train[i], i)
        print("Done training with the ", i + 1, "th data. On to the ", i + 2,"th data")

    neuralNet.test(x_test[23])
    print("The desired output of the test is: ", y_test[23])
