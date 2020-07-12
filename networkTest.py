import numpy as np
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits import mplot3d

class network:

    def __init__(self, layer1Weights, layer2Weights, layer3Weights, layer1Biases, layer2Biases, layer3Biases):

        self.layer1Weights = layer1Weights
        self.layer2Weights = layer2Weights
        self.layer3Weights = layer3Weights

        self.layer1Biases = layer1Biases
        self.layer2Biases = layer2Biases
        self.layer3Biases = layer3Biases

    def sigmoid(x):

        output = 1 / (1 + np.exp(-x))

        return output

    def sigmoid2(self, x):

        output = np.zeros((1, 784))

        for i in range(len(x)):

            output[0][i] = 1/(1+np.exp(-(x[i])))

        return output

    def think(self, trainingSetInputs):

        outputFromLayer1 = self.sigmoid(np.dot(trainingSetInputs, self.layer1Weights) - self.layer1Biases)
        outputFromLayer2 = self.sigmoid(np.dot(outputFromLayer1, self.layer2Weights) - self.layer2Biases)
        outputFromLayer3 = self.sigmoid(np.dot(outputFromLayer2, self.layer3Weights) - self.layer3Biases)

        return outputFromLayer3

if __name__ == "__main__":

    np.random.seed(1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    
