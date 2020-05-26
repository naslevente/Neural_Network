#import tensorflow_datasets as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

class neuronLayer():

    def __init__(self, numOfNeurons, numOfInputsPerNeuron):

        #Each neuron layer is comprised of 2 matrices: one is the weights and the other is the biases
        #Initially, I just choose random numbers which will output random results
        self.synaptic_weights = np.random.random((numOfInputsPerNeuron, numOfNeurons)) - 0.5
        self.biases = np.random.random((1, numOfNeurons)) - 0.5

class neuralNetwork():

    #This is the contrustor of the class. "self" is essentially the super keyword from java
    def __init__(self, layer1, layer2, layer3):

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    #input = np.linspace(-10, 10, 100)

    #I used the sigmoid function as my activation function
    #There are multiple sigmoids because some take in a matrix and return a sigmoided matrix and the other one just takes one value
    def sigmoid3(self, y):

        return 1/(1+np.exp(-(y)))

    def sigmoid(self, x):

        output = np.zeros((1, 16))

        for i in range(x.size):

            output[0][i] = 1/(1+np.exp(-(x[0][i])))

        return output

    def sigmoid2(self, x):

        output = np.zeros((1, 784))

        for i in range(x.size):

            output[0][i] = (1/(1+np.exp(-((x[i] - 112.5) * (1 / 25.5)))))

        return output

    #Sigmoid derivative needed for backpropagation algorithm
    def sigmoidDer(self, x):

        output = (1/(1 + np.exp(x))) * ( 1 - (1/(1 + np.exp(-(x)))))

        return output

    #ignore
    def function(self, x, y):

        return (self.sigmoid3(y) - x) ** 2

    #from matplotlib import pyplot as plt
    #plt.plot(input, sigmoid(input), c="r")
    #plt.show()

    #This is the train method. I set a learning rate for both the bias and weights
    def train(self, input_pixels, desired_output):

        print(self.layer1.synaptic_weights)
        print(self.layer2.synaptic_weights)
        print(self.layer3.synaptic_weights)

        #learningRateWeight = 0.6
        #learningRateBias = 0.4

        input = []

        for i in range(0, 28):

            input = np.concatenate((input, input_pixels[i]))

        #I break up the input to separate the input and output from the input line from the text file
        #An example input looks like this: 21 90 63-0 0 0 1 0 0
        #The first 3 numbers are the rgb values and the numbers to the right of the rgb values is the desired output (this example means that the output neuron layer should say "0 red 0 orange 0 yellow 1 green 0 blue 0 white" which means the color is green)

        #This is where I send the inputs to the think method which computes the output through matrix multiplication
        #I record the output from each layer (obviously the output will be a number between 0 and 1 because it is sigmoided)
        outputFromLayer1, outputFromLayer2, outputFromLayer3 = self.think(input)
        trainingSetOutputs = [0] * 10
        trainingSetOutputs[desired_output] = 1

        #After getting the outputs, we can backpropagate (math is commented at the method itself)
        layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge = self.weightDer(outputFromLayer3, outputFromLayer2, outputFromLayer1, trainingSetOutputs, input)

        print("Weight Nudges: ")
        print(layer3WeightNudges)
        print(layer2WeightNudges)
        print(layer1WeightNudges)
        print("done")

        '''
        self.layer1.synaptic_weights -= (layer1WeightNudges)
        self.layer2.synaptic_weights -= (layer2WeightNudges)
        self.layer3.synaptic_weights -= (layer3WeightNudges)

        self.layer1.biases -= (layer1BiasNudge)
        self.layer2.biases -= (layer2BiasNudge)
        self.layer3.biases -= (layer3BiasNudge)
        '''

        #The idea is to average the nudges through all the training examples so that the network doesn't learn to recognize one specific example
        '''
        layer1WeightAverage += layer1WeightNudges
        layer2WeightAverage += layer2WeightNudges
        layer3WeightAverage += layer3WeightNudges

        layer1BiasAverage += layer1BiasNudge
        layer2BiasAverage += layer2BiasNudge
        layer3BiasAverage += layer3BiasNudge
        '''

        self.layer1.synaptic_weights -= (layer1WeightNudges)
        self.layer2.synaptic_weights -= (layer2WeightNudges)
        self.layer3.synaptic_weights -= (layer3WeightNudges)

        self.layer1.biases -= (layer1BiasNudge)
        self.layer2.biases -= (layer2BiasNudge)
        self.layer3.biases -= (layer3BiasNudge)

    #And now for the meat of the network
    def weightDer(self, outputFromLayer3, outputFromLayer2, outputFromLayer1, desiredOutput, trainingSetInputs):

        print("CHECK: ", len(trainingSetInputs))
        #I create empty 1 X 6 matrices to record the nudges after I compute them
        layer3WeightNudges = np.zeros((16, 10))
        layer2WeightNudges = np.zeros((16, 16))
        layer1WeightNudges = np.zeros((784, 16))
        layer2ActivationNudge = np.zeros((1, 16))
        layer1ActivationNudge = np.zeros((1, 16))
        layer0ActivationNudge = np.zeros((1, 784))

        layer1BiasNudge = np.zeros((1, 16))
        layer2BiasNudge = np.zeros((1, 16))
        layer3BiasNudge = np.zeros((1, 10))

        layer1Bias = True;
        layer2Bias = True;
        layer3Bias = True;

        len1 = 10
        len2 = 16
        len3 = 16
        inputs = self.sigmoid2(trainingSetInputs)

        for i in range(0, 784):

            for x in range(0, len3):

                for y in range(0, len2):

                    for t in range(0, len1):

                        if layer3Bias:

                            layer3BiasNudge[0][t] = 2 * (outputFromLayer3[0][t] - desiredOutput[t]) * self.sigmoidDer(outputFromLayer3[0][t])

                        if t == len1 - 1 and y == 0:

                            layer1Bias = False

                        layer3WeightNudges[y][t] = outputFromLayer2[0][y] * 2 * (outputFromLayer3[0][t] - desiredOutput[t]) * self.sigmoidDer(outputFromLayer3[0][t])
                        layer2ActivationNudge[0][y] += 2 * (outputFromLayer3[0][t] - desiredOutput[t]) * self.sigmoidDer(outputFromLayer3[0][t]) * self.layer3.synaptic_weights[y][t]

                    if layer2Bias:

                        layer2BiasNudge[0][y] = layer2ActivationNudge[0][y] * self.sigmoidDer(outputFromLayer2[0][y])

                    if y == len2 - 1 and x == 0:

                        layer2Bias = False

                    layer2WeightNudges[x][y] = layer2ActivationNudge[0][y] * self.sigmoidDer(outputFromLayer2[0][y]) * outputFromLayer1[0][x]
                    layer1ActivationNudge[0][x] += layer2ActivationNudge[0][y] * self.sigmoidDer(outputFromLayer2[0][y]) * self.layer2.synaptic_weights[x][y]

                len1 = 0

                if layer1Bias:

                    layer1BiasNudge[0][x] = layer1ActivationNudge[0][x] * self.sigmoidDer(outputFromLayer1[0][x])

                if x == len3 - 1 and i == 0:

                    layer1Bias = False

                layer1WeightNudges[i][x] = layer1ActivationNudge[0][x] * self.sigmoidDer(outputFromLayer1[0][x]) * inputs[0][i]
                #layer0ActivationNudge[0][i] += layer1ActivationNudge[0][x] * self.sigmoidDer(outputFromLayer1[0][x]) * self.layer1.synaptic_weights[i][x]

            len2 = 0

        return layer3WeightNudges, layer2WeightNudges, layer1WeightNudges, layer3BiasNudge, layer2BiasNudge, layer1BiasNudge

    #This thinking part is just multiplying the matrices to get the output layer. I didn't find a matrix mutliplication method that suits my needs so I just wrote it but there may be one out there
    def think(self, inputs):

        outputFromLayer1 = self.sigmoid(np.dot(self.sigmoid2(inputs), self.layer1.synaptic_weights) - self.layer1.biases)
        outputFromLayer2 = self.sigmoid(np.dot(outputFromLayer1, self.layer2.synaptic_weights) - self.layer2.biases)
        outputFromLayer3 = self.sigmoid(np.dot(outputFromLayer2, self.layer3.synaptic_weights) - self.layer3.biases)

        return outputFromLayer1, outputFromLayer2, outputFromLayer3

    def matrixMultiply(self, matrix1, matrix2):

        output = np.zeros((1, 16))

        for i in range(len(matrix1)):

            for j in range(len(matrix2[0])):

                for k in range(len(matrix2)):

                    output[0][j] += matrix1[i][k] * matrix2[k][j]

        return output

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

    '''
    image_index = 3829 # You may select anything up to 60,000
    print(y_train[image_index]) # The label is 8
    print(x_train[image_index][6])
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()
    '''

    layer1 = neuronLayer(16, 784)
    layer2 = neuronLayer(16, 16)
    layer3 = neuronLayer(10, 16)

    neuralNet = neuralNetwork(layer1, layer2, layer3)

    for i in range(50000):

        neuralNet.train(x_train[i], y_train[i])
        print("Done training with the ", i + 1, "th data. On to the ", i + 2,"th data")

    neuralNet.test(x_test[23])
    print("The desired output of the test is: ", y_test[23])
