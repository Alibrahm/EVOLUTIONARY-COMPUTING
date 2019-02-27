#!/usr/bin/python3

'''
P15/37272/2016
IBRAHIM ALI
Machine Learning Algorithms
Not using data output
'''

import sys

# Asrgparse obsoletes the need for usage()
# Using RegEx
# Principle Component Analysis proviced by numpy
import argparse as argpa
import math
import numpy as nump
import pandas as pand
import random as rand
import re
import textwrap

import classes

import matplotlib.pyplot as mplt
from numpy import linalg as nlg

from matplotlib import style
style.use('ggplot')

# Return statuses
# Inumput file not specified
# FNOT_SPECIFIED_R = 3
INVALID_FILE_CONTENTS = 4


# Using Principle Compom=nent Analysis to generate
# Human plottable data
# SO
def PCA(data, dims_rescaled_data=2):
    '''
    Return data transformed to N dimendions
    Return original data form 2D NumPy array
    '''
    # m_dimensions, n_dimensions = data.shape

    # mean center the data
    data -= data.mean(axis=0)

    # calculate the covariance matrix
    R = nump.cov(data, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evigen_vals, eigen_vects = nlg.eigh(R)

    # Sort eigenvalue in decreasing order
    idx = nump.argsort(evigen_vals)[::-1]

    eigen_vects = eigen_vects[:, idx]

    # sort eigenvectors according to same index
    evigen_vals = evigen_vals[idx]

    # Select the first N eigenvectors (N is desired dimension
    # of rescaled data array) or dims_rescaled_data
    eigen_vects = eigen_vects[:, :dims_rescaled_data]

    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return nump.dot(eigen_vects.T, data.T).T, evigen_vals, eigen_vects
# End PCA


def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    null, null, eigen_vects = PCA(data, dim_rescaled_data=2)

    m_dimensions, n_dimensions = data.shape

    data_recovered = nump.dot(eigen_vects, m_dimensions).T
    data_recovered += data_recovered.mean(axis=0)

    assert nump.allclose(data, data_recovered)
# End test_PCA


def plot_pca(data):
    clr1 = '#2026B2'
    fig = mplt.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    mplt.show()
# End plot_pca
# End SO


def check_algo(data_set, args):
    if(args.algorithm == 0):
        print('[+] K-Means Algorithm')

        if (data_set is None):
            data_set = nump.random.random_integers(1, 50, (30, 2))

        kmeans_algo(data_set, args)
    elif(args.algorithm == 1):
        print('[+] Back Propagation Algorithm')

        if (data_set is None):
            data_set = nump.array([
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1]])

        backp_algo(data_set, args)
    elif(args.algorithm == 2):
        print('[+] K-Nearest Neighbours Algorithm')

        if (data_set is None):
            data_set = nump.random.random_integers(1, 50, (50, 2))
            '''
            data_set = nump.array([

                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)],
                [rand.randrange(0, 10), rand.randrange(0, 10)]])
            '''
        knn_algo(data_set, args)
    elif(args.algorithm == 3):
        print('[+] Perceptron Algorithm')

        if (data_set is None):
            data_set = nump.array([
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1]])

        percept_algo(data_set, args)
    elif(args.algorithm == 4):
        print('[+] Self-Organizing Map Algorithm')

        data_set = nump.random.random_integers(1, 50, (50, 3))
        som_algo(data_set, args)
# End check_algo


def split_set(data_set, args):
    if(args.training_scheme == 0):
        # 90:10
        training_size = int(len(data_set) * 0.9)
    elif(args.training_scheme == 1):
        # 80:20
        training_size = int(len(data_set) * 0.8)
    elif(args.training_scheme == 2):
        # 70:30
        training_size = int(len(data_set) * 0.7)
    # End if

    '''
    validate_size = int(len(data_set) - training_size)

    training_set = nump.array_split(data_set, training_size)
    validation_set = nump.array_split(
        data_set,
        int(len(data_set) - training_size))
    '''

    training_set = data_set[:training_size]
    validation_set = data_set[training_size:]

    return {
        'training_set': training_set,
        'validation_set': validation_set}
# End split_set


def knn_algo(data_set, args):
    # Compulsory
    sets = split_set(data_set, args)
    training_set = sets['training_set']
    validation_set = sets['validation_set']

    if (args.clusters >= len(training_set)):
        print("[!] Impractical cluster size\n")
        exit(1)

    knn_cluster = classes.KNNCluster(args.clusters)
    output = knn_cluster.query_output(training_set, rand.randrange(0, 10))

    print("[+] Result: " + str(output) + "\n")

# End knn_algo


def backp_algo(data_set, args):
    # Compulsory
    sets = split_set(data_set, args)
    training_set = sets['training_set']
    validation_set = sets['validation_set']

    training_set_input, expected_output = training_set[:, :-1], training_set[:, -1]

    # Initialize to number of columns (input nodes)
    backp_nn = classes.Neural_Net(nump.size(training_set_input, 1))

    # Train 0 - 1000 times
    #for i in (rand.randrange(0, 1000):

    # Train using all training data
    for i in range(len(training_set_input)):
        error_rate = 1

        while error_rate > 0.1:
            backp_nn.expected_output = expected_output[i]
            backp_nn.train(training_set_input[i], expected_output[i])

            # Calculate the mean sum squared loss
            sq_error = backp_nn.squared_error(backp_nn.expected_output, backp_nn.output)
            error_rate = (1/ len(training_set_input)) * sq_error

            print(
                "Input: \n" + str(training_set[i]) +
                "\n\nActual Output: \n" + str(backp_nn.expected_output) +
                "\n\nPredicted Output: \n" + str(backp_nn.output) +
                "\n\nError rate: " + str(error_rate) + "\n")
        # End while

# End backp_algo


def som_algo(data_set, args):
    # Compulsory
    sets = split_set(data_set, args)
    training_set = sets['training_set']
    validation_set = sets['validation_set']

    koh_som = classes.KSOM(nump.size(training_set, 0), nump.size(training_set, 1))
    koh_som.organize_data(training_set)

    print("[+] Final map: \n\t" + str(koh_som.input))
# End som_algo


def percept_algo(data_set, args):
    sets = split_set(data_set, args)
    training_set = sets['training_set']
    validation_set = sets['validation_set']

    training_set_input, expected_output = training_set[:, :-1], training_set[:, -1]

    perceptr = classes.Perceptron(nump.size(training_set_input, 1))

    for i in range(len(training_set_input)):
        error = 1

        while error > 0.1:
            perceptr.expected_output = expected_output[i]
            perceptr.generate_output(training_set_input[i])

            error = perceptr.error(perceptr.expected_output, perceptr.output)

            if (perceptr.iteration_limit == 0):
                break
            perceptr.iteration_limit -= 1
        # End while

            print(
                "Input: \n" + str(training_set[i]) +
                "\n\nActual Output: \n" + str(perceptr.expected_output) +
                "\n\nPredicted Output: \n" + str(perceptr.output) +
                "\n\nError: " + str(error) + "\n")
        # End while
# End percept_algo


def kmeans_algo(data_set, args):

    sets = split_set(data_set, args)
    training_set = sets['training_set']
    validation_set = sets['validation_set']

    '''
    print(
        "Train:" +
        str(len(training_set)) +
        " Validate:" +
        str(len(validation_set)))  # Test

    centroids = []
    index = 0

    while index < args.clusters:
        # centroids.append(nump.take(training_set, index))
        centroids.append(training_set.iloc[index, :].values)

        index += 1
    # End while

    centroids = data_set.head(args.clusters)

    # Drop 'y' column
    # Perform PCA on resulting set
    # Create a plottable matrix
    num_columns = training_set.get_dtype_counts()
    y_column = training_set.iloc[:, -1].values.tolist()

    training_set.drop(data_set.columns[num_columns - 1], axis=1)
    training_set, evigen_vals, eigen_vects = PCA(training_set, 1)

    #training_set = pand.Series(training_set)
    training_set['Yval'] = pand.Series(y_column)

    print(training_set)
    # print(centroids)
    '''

    if (args.clusters >= len(training_set)):
        print("[!] Impractical cluster size\n")
        exit(1)

    kmeans_cluster = classes.KMClusters(args.clusters)
    kmeans_cluster.cluster_data(training_set)

    print("[+] Final clusters: \n" + str(kmeans_cluster.clusters) + "\n")
    print("[+] Plotting results")

    colours = ["b", "y", "b", "darkgreen", "c", "m", "lime"]
    mplt.title("K-Means Clustering")

    for i in kmeans_cluster.clusters:
        for plot_coord in kmeans_cluster.clusters[i]:
            x_val = plot_coord[0]
            y_val = plot_coord[-1]

            mplt.scatter(x_val, y_val, c=colours[i])

        x_val = kmeans_cluster.centroids[i][0]
        y_val = kmeans_cluster.centroids[i][-1]

        mplt.scatter(x_val, y_val, c=colours[i], marker="*")

    mplt.savefig("kmeans.png")
    mplt.show()

# End kmeans_algo


def main(sys_args):
    # global FNOT_SPECIFIED_R
    global INVALID_FILE_CONTENTS

    # To pass raw text for help generation
    # formatter_class=argpa.RawTextHelpFormatter,
    parser = argpa.ArgumentParser(
        description='Machine learning algorithms.',
        formatter_class=argpa.RawTextHelpFormatter,
        epilog='P15/35280/2015')

    # Allow multiple inumput files
    parser.add_argument(
        '-f',
        '--file',
        action='append',
        required=False,
        help='input file(s) containing training data')

    # Removing indentations with textwrap
    parser.add_argument(
        '-s',
        '--training-scheme',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=textwrap.dedent('''\
        specify [training:validation] scheme for the algorithm
        [0] 90:10
        [1] 80:20
        [2] 70:30'''))

    parser.add_argument(
        '-k',
        '--clusters',
        type=int,
        default=4,
        help="number of clusters to create")

    parser.add_argument(
        '-t',
        '--test',
        action='append',
        help='inumput file(s) containing the test data')

    parser.add_argument(
        '-a',
        '--algorithm',
        required=True,
        type=int,
        choices=[0, 1, 2, 3, 4],
        help=textwrap.dedent('''\
        specify agorithm to execute
        [0] K-Means Clustering
        [1] Back Propagation algorithm
        [2] K-Nearest Neighbours
        [3] Perceptron
        [4] Self-Organizing Maps'''))

    args = parser.parse_args()

    # Made obsolete by required='True'
    '''
    if(args.file is None):
        print("[!] Inumput file(s) not specified")
        parser.print_help()

        exit(FNOT_SPECIFIED_R)
    # End if
    '''

    comma_re = re.compile("[0-9.]*,[0-9.]*")
    semi_re = re.compile("[0-9.]*;[0-9.]*")
    space_re = re.compile("[0-9.]* [0-9.]*")
    tab_re = re.compile("[0-9.]*\t[0-9.]*")

    if(args.file is not None):
        while(len(args.file) > 0):
            param_delim = None
            counter = 0

            print("\n[+] Operating on \"" + str(args.file[0]) + "\" dataset")

            # Handle comma, space, semicolon & tab separated lists
            with open(args.file[0]) as data_file:
                for data_line in data_file:
                    if(comma_re.match(data_line) and param_delim is not "Comma"):
                        param_delim = "Comma"
                    elif(semi_re.match(data_line) and param_delim is not "Semi"):
                        param_delim = "Semi"
                    elif(space_re.match(data_line) and param_delim is not "Space"):
                        param_delim = "Space"
                    elif(tab_re.match(data_line) and param_delim is not "Tab"):
                        param_delim = "Tab"
                    else:
                        if(counter >= 5):
                            data_file.close()
                            break
                        # End if
                    counter += 1
                # End for
            # End with

            if(param_delim is "Comma"):
                data_set = pand.read_csv(args.file[0])

                check_algo(data_set.astype(float), args)
            elif(param_delim is "Semi"):
                data_set = pand.read_csv(args.file[0], header=None, delimiter=";")

                check_algo(data_set.astype(float), args)
            elif(param_delim is "Space"):
                data_set = pand.read_csv(args.file[0], header=None, delimiter=" ")

                check_algo(data_set.astype(float), args)
            elif(param_delim is "Tab"):
                data_set = pand.read_csv(args.file[0], header=None, delimiter="\t")

                check_algo(data_set.astype(float), args)
            else:
                print(
                    "[!] Abiguous value separator in \"" +
                    str(args.file[0]) +
                    "\"")

            args.file = args.file[1:]
        # End while
    else:
        print("\n[+] Operating on default dataset")

        data_set = None
        check_algo(data_set, args)
    # End if

# End main


if(__name__ == "__main__"):
    '''
    if (len(sys.argv) < 2):
        usage()

        sys.exit(1)  # Exit due to invalid argument number
    # End if
    '''

    main(sys.argv)
