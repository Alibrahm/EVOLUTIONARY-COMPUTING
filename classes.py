import numpy as nump
import pandas as pand
import random as rand
import operator

from numpy import linalg as nlg


class Neural_Net():
    def __init__(self, len_data):
        self.input_layer_size = len_data
        self.output_layer_size = 1
        self.hidden_layer_size = len_data
        self.learning_rate = float(rand.randrange(0, 10) / 10)
        self.epoch = 0

        self.output_1 = None
        self.output_2 = None
        self.output_3 = None
        self.output = None
        self.output_error = None
        self.output_delta = None
        self.output_3_delta = None

        # Weight matrix from input to hidden layer
        # (I x H)
        self.weight_1 = nump.random.randn(self.input_layer_size,
            self.hidden_layer_size)

        # Weight matrix from hidden to output layer
        # (H x O)
        self.weight_2 = nump.random.randn(self.hidden_layer_size,
            self.output_layer_size)
    # End __init__

    def forward(self, X):
        # Forward propagation through our network

        # Input layer
        # Dot product of summed input and first set of weights`
        #summed_up_input = X.sum(axis=0)
        self.output_1 = nump.dot(self.weight_1.T, X)
        print("[+] O1: \n\t", self.output_1)

        self.output_2 = self.activation_func(self.output_1)
        print("[+] O2: \n\t", self.output_2)

        # Hidden layer
        # Dot product of hidden layer output and second set of weights`
        self.output_3 = nump.dot(self.weight_2.T, self.output_2)
        print("[+] O3: \n\t", self.output_3)

        # Output layer
        # Activation function to generate final output
        self.output = self.activation_func(self.output_3)
        print("[+] Y: \n\t", self.output)

        return self.output
    # End forward

    def activation_func(self, outp):
        # Sigmoid function
        return (1 / (1 + nump.exp(-outp)))
        #math.tanh(x) # Doesm't work
    # End activation_func

    def error_func(self, outp):
        # (yt - ys ) * ys * (1 - ys)
        # Error calculated out of this function
        return outp * (1 - outp)
    # End error_func

    def backward(self, X, y, algo_outp):
        # Backward propgate through the network

        # Get output layer delta
        # Expected - Actual)
        self.output_error = y - algo_outp
        self.output_delta = self.output_error * self.error_func(algo_outp)

        weight_2_delta = (self.learning_rate * self.output_delta)
        self.weight_2 += weight_2_delta

        # Get hidden layer deltas
        # e = W0 . d
        # d* = [a x g x (1-g)]
        #      [b x h x (1-h)]

        e = self.weight_2.dot(self.output_delta)

        # e x [g]
        #     [h]
        egh = e * self.output_3
        self.output_3_delta = egh * (1 - self.output_3)

        '''
        self.output_2_error = self.output_delta.dot(self.output_2)

        self.output_2_delta = self.output_2_error * self.error_func(self.output_2)

        self.weight_1 += nump.dot(X, weight_1_delta.T)'''

        weight_1_delta = (self.learning_rate * self.output_3_delta)

        self.weight_1 += weight_1_delta.T
    # End backward

    def train(self, X, y):
        print("[+] Epoch " + str(self.epoch))

        algo_outp = self.forward(X)
        self.backward(X, y, algo_outp)

        self.epoch += 1
    # End train

    def squared_error(self, t, y):
        return (t - y) ** 2
    # End squared_error
# End class Neural_Net


class Perceptron():
    def __init__(self, len_data):
        self.input_size = len_data
        self.output_size = 1
        self.learning_rate = float(rand.randrange(0, 10) / 10)
        self.iteration_limit = self.input_size ** 2
        self.epoch = 0

        self.output_1 = None
        self.output = None
        self.output_error = None
        self.output_delta = None

        # Weight matrix
        self.weight_1 = nump.random.randn(self.input_size, self.output_size)
    # End __init__

    def activation_func(self, outp):
        # Sigmoid function
        return (1 / (1 + nump.exp(-outp)))
        #math.tanh(x) # Doesm't work
    # End activation_func

    def error_func(self, outp):
        # (yt - ys ) *ys * (1 - ys)
        # Error calculated out of this function
        return outp * (1 - outp)
    # End error_func

    def error(self, t, y):
        return (t - y)
    # End error

    def generate_output(self, X):
        print("[+] Epoch " + str(self.epoch))

        self.output_1 = nump.dot(self.weight_1.T, X)

        self.output = self.activation_func(self.output_1)

        self.output_error = self.output - self.expected_output
        self.output_delta = self.output_error * self.error_func(self.output)

        weight_1_delta = (self.learning_rate * self.output_delta)
        self.weight_1 += weight_1_delta

        self.epoch += 1
        return self.output
    # End generate_output
# End Perceptron


class KMClusters():
    def __init__(self, size):
        self.num_clusters = size
        self.centroids = []
        self.clusters = {}
        self.old_centroids = {}
        self.clustering_activation = 50
        self.epoch = 0
    # End __init__

    def cluster_data(self, X):
        for i in range(self.num_clusters):
            self.centroids.append(X[i])
            self.clusters[i] = []
            self.cluster_enable = 1

        for coordinate in X:
            distances = []

            for e in range(self.num_clusters):
                distances.append(nlg.norm(self.centroids[e] - coordinate))

            cluster_loc = distances.index(min(distances))

            self.clusters[cluster_loc].append(coordinate)

        self.clustering_timeout = X.size ** 2

        stability = 0
        adjusted = True
        while True:
            self.clustering_timeout -= 1
            counter = 0
            for cluster in self.clusters:
                self.old_centroids = self.centroids
                self.centroids[counter] = nump.average(self.clusters[cluster], axis=0)
                counter += 1

            if (self.clustering_timeout == 0 or stability == (X.size * self.num_clusters)):
                break

            print("[+] centroids: \n" + str(self.centroids))

            if (adjusted is False):
                stability += 1
            else:
                stability = 0

            adjusted = True

            for i in range(self.clustering_timeout):
                if (adjusted is False):
                    break

                print("[+] Epoch " + str(self.epoch) + "\n")

                adjusted = False
                located = None

                current_cluster = 0
                for cluster in self.clusters:
                    for coordinate in self.clusters[cluster]:
                        distances = []
                        for e in range(self.num_clusters):
                            distances.append(nlg.norm(self.centroids[e] - coordinate))

                        cluster_loc = distances.index(min(distances))
                        foreign_presence = (self.clusters[cluster_loc] == coordinate).all(axis=1)

                        # nump.where((self.clusters[cluster_loc][:] == /coordinate).all(axis=1))
                        foreign_indices = nump.where(foreign_presence == (True))

                        # Move coordinate to new cluster
                        if (foreign_indices[0].size == 0):
                            local_presence = (self.clusters[current_cluster] == coordinate).all(axis=1)
                            local_indices = nump.where(local_presence == True)

                            if (local_indices[0].size > 0):
                                for i in local_indices:
                                    nump.delete(self.clusters[current_cluster], (i[0]))
                                    self.clusters[cluster_loc].append(coordinate)

                                print("[+] Cluster adjusted")
                                adjusted = True
                    current_cluster += 1

                print("[+] Clusters: \n" + str(self.clusters) + "\n")
                self.epoch += 1
                for cluster in self.clusters:
                    self.centroids[cluster] = nump.average(self.clusters[cluster])
    # End cluster_data
# End class KMClusters


class KNNCluster():
    def __init__(self, count):
        self.num_neighbours = count
        self.neighbours = {}
    # End __init__

    def get_neighbours(self, X, query_instance):
        distances = []

        for value in X:
            distances.append(nump.array([int(nlg.norm(query_instance - value[0])), value[-1]]))

        distances = nump.array(sorted(distances, key=operator.itemgetter(0)))
        self.neighbours = distances[:self.num_neighbours]

        print("[+] Neighbours: \n" + str(self.neighbours) + "\n")

        return nump.array(self.neighbours[:, -1])
    # End get_neighbours

    def query_output(self, X, query_instance):
        output = nump.average(self.get_neighbours(X, query_instance), axis=0)
        return output
    # End query_output
# End KNCluster


class KSOM():
    def __init__(self, data_size, data_elements):
        self.data_amnt = data_size
        self.data_dimensions = data_elements
        # neighbourhood_func ensures only 2 surrounding weights are adjusted.
        # self.radius = rand.randrange(0, 100) # unused
        self.learning_rate = float(rand.randrange(0, 10) / 10)
        self.neighbourhood_const = float(0.5)
        self.epoch = 0

        self.weights = nump.random.randn(self.data_amnt, self.data_dimensions)
        self.weights_delta = None
        self.iteration_limit = self.data_amnt ** 2
    # End __init__

    def neighbourhood_func(self, i, k):
        if (i == k):
            result = 1
        elif(i == (k + 1) or i == (k - 1)):
            result = self.neighbourhood_const
        else:
            result = 0

        return result
    # End neighbourhood_func

    def organize_data(self, X):
        self.input = X

        while self.iteration_limit > 0:
            print("[+] Epoch: " + str(self.epoch))

            self.train_som()

            #print("[+] Output: \n\t" + str(self.input))
            self.iteration_limit -= 1
            self.epoch += 1
        # End while
    # End organize_data

    def train_som(self):
        selection = rand.randrange(0, nump.size(self.input, 0))
        train_vector = self.input[selection]

        distances = []
        for weight in self.weights:
            distances.append(nlg.norm(weight - train_vector))

        # k with maximum excitation
        k = distances.index(min(distances))

        counter = 0
        for weight in self.weights:
            self.weights_delta = self.learning_rate * self.neighbourhood_func(counter, k) * (train_vector - weight)

            self.weights[counter] = weight + self.weights_delta
        # Adjust neighbourhood function
        self.neighbourhood_const -= float(0.01)
    # End train_som
# End KSOM
