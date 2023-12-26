#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'so':1, 'touching':1, 'quite':0, 'impressive':0, 'not':-1, 'boring':-1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    word_list = x.split(' ')
    word_dict = {}
    for word in word_list:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    
    return word_dict
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    
    for _ in range(numIters):
        for (x, y) in trainExamples:
            feature = featureExtractor(x)
            if (y==1):
                increment(weights, eta*(1-sigmoid(dotProduct(weights, feature))), feature)
            elif (y==-1):
                increment(weights, -eta*sigmoid(dotProduct(weights, feature)), feature)
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    word_list = x.split(' ')
    phi = {}
    for i in range(len(word_list)-n+1):
        n_words = ' '.join(word_list[i:n+i])
        if n_words in phi:
            phi[n_words] += 1
        else:
            phi[n_words] = 1
    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    def get_distance(a, b):
        x = (a[0]-b[0])**2
        y = (a[1]-b[1])**2
        return x+y

    trains = [[-1, 0], [2, 0], [0, 3], [4, 3]]
    centroids = [[-2, 0], [3, 0]]

    while(True): 
        cluster = []
        #각 값에 center값 할당
        for point in trains:
            distance_from_centroids = []
            for centroid in centroids:
                distance_from_centroids.append(get_distance(point, centroid))
            cluster_index = distance_from_centroids.index(min(distance_from_centroids))
            cluster.append(cluster_index)

        print(centroids)
        #cluster에 따른 centroid재설정
        pre_centroids = centroids
        centroids = [[0 for i in range(2)] for j in range(2)]
        centroid_counter = [0 for i in range(2)]
        for point, centroid in zip(trains, cluster):
            centroids[centroid][0] += point[0]
            centroids[centroid][1] += point[1]
            centroid_counter[centroid] += 1

        for i in range(len(centroids)):
            if centroid_counter[i] == 0:
                continue
            centroids[i][0] /= centroid_counter[i]
            centroids[i][1] /= centroid_counter[i]

        if(pre_centroids == centroids):
            break

    #결과 centroids dictionary에 담기
    result_centroids = []
    for centroid in centroids:
        dic = {'mu_x': centroid[0], 'mu_y': centroid[1]}
        result_centroids.append(dic)
    
    return result_centroids
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    def get_distance(a, b):
        x = (a[0]-b[0])**2
        y = (a[1]-b[1])**2
        return x+y

    trains = [[-1, 0], [2, 0], [0, 3], [4, 3]]
    centroids = [[-1, -1], [2, 3]]

    while(True): 
        cluster = []
        #각 값에 center값 할당
        for point in trains:
            distance_from_centroids = []
            for centroid in centroids:
                distance_from_centroids.append(get_distance(point, centroid))
            cluster_index = distance_from_centroids.index(min(distance_from_centroids))
            cluster.append(cluster_index)

        print(centroids)
        #cluster에 따른 centroid재설정
        pre_centroids = centroids
        centroids = [[0 for i in range(2)] for j in range(2)]
        centroid_counter = [0 for i in range(2)]
        for point, centroid in zip(trains, cluster):
            centroids[centroid][0] += point[0]
            centroids[centroid][1] += point[1]
            centroid_counter[centroid] += 1

        for i in range(len(centroids)):
            if centroid_counter[i] == 0:
                continue
            centroids[i][0] /= centroid_counter[i]
            centroids[i][1] /= centroid_counter[i]

        if(pre_centroids == centroids):
            break

    #결과 centroids dictionary에 담기
    result_centroids = []
    for centroid in centroids:
        dic = {'mu_x': centroid[0], 'mu_y': centroid[1]}
        result_centroids.append(dic)
    
    return result_centroids
    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    def find_distance(d1, d2):
        result = 0
        for k, v in d2.items():
            result += (d1.get(k, 0) - v) ** 2
        return result
    # initialize centroids randomly
    centers = random.sample(examples, K)

    prevLoss = None
    for i in range(maxIters):
        # assign each example to the nearest centroid
        assignments = []
        loss = 0
        for example in examples:
            distances = [find_distance(example, center) for center in centers]
            minDistIndex = distances.index(min(distances))
            assignments.append(minDistIndex)
            loss += distances[minDistIndex]**2
        if loss == prevLoss:
            break
        prevLoss = loss

        # recompute centroids
        for j in range(K):
            clusterExamples = [examples[idx] for idx in range(len(examples)) if assignments[idx] == j]
            if clusterExamples:
                centers[j] = Counter()
                for example in clusterExamples:
                    centers[j].update(example)
                centers[j] = {k: centers[j][k] / len(clusterExamples) for k in centers[j]}

    return centers, assignments, loss


    # END_YOUR_ANSWER

