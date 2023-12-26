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
    return [{'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3.0, 'mu_y': 1.5}]

def problem_3a_2():
    return [{'mu_x': -0.5, 'mu_y': 1.5}, {'mu_x': 3.0, 'mu_y': 1.5}]


############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    # x1은 center, x2는 example로 설정함. 모든 key가 아닌 x2의 key들만 고려하도록 해서 시간 단축
    def find_distance(d1, d2):
        result = 0
        for k, v in d2.items():
            result += (d1.get(k, 0) - v) ** 2
        return result
 
    centers = random.sample(examples, K)

    pre_loss = 0
    # examples들 center에 assign
    for i in range(maxIters):
        assignments = []
        loss = 0
        for example in examples:
            distances = [find_distance(center, example) for center in centers]
            min_index = distances.index(min(distances))
            assignments.append(min_index)
            loss += distances[min_index]
        # 수렴했다면 break
        if loss == pre_loss:
            break
        pre_loss = loss

        # center들 재설정
        new_centers = []
        for j in range(K):
            new_center = Counter()
            examples_in_center = [examples[index] for index in range(len(examples)) if assignments[index] == j]
            for example in examples_in_center:
                new_center.update(example)
            new_center = {k: new_center[k] / len(examples_in_center) for k in new_center}
            new_centers.append(new_center)
            
        centers = new_centers
    return centers, assignments, loss


