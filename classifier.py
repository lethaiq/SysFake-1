"""
Module Docstring Placeholder
import os
import time
import random
"""

import numpy as np
import sklearn

from sklearn.model_selection import KFold

from feature_extraction import ArticleVector

#Following is somewhat unnecessary, but may be useful if we ever separate the data.

#each filename should be a file containing article urls separated by spaces.
TRAINING_FILE_DICT = {'real_news_urls-training.txt' : 1,
                      'fake_news_urls-training.txt' : 2,
                      'opinion_urls-training.txt' : 3,
                      'polarized_news_urls-training.txt' : 5,
                      'satire_urls-training.txt' : 7}

TESTING_FILE_DICT = {'real_news_urls-testing.txt' : 1,
                     'fake_news_urls-testing.txt' : 2,
                     'opinion_urls-testing.txt' : 3,
                     'polarized_news_urls-testing.txt' : 5,
                     'satire_urls-testing.txt' : 7}

def extract_data(filename, label):
    """
    Method Docstring Placeholder
    """
    data = extract_urls(filename)
    data_x = []
    count = 0
    for url in data[400: 800]:
        count += 1
        print('Current url:', url, '|| Visited', count, 'websites...')
        try:
            data_x.append(ArticleVector(url).vector)
        except: #pylint: disable=W0702
            print('IT FAILED')
            continue
    data_y = [label] * len(data_x)
    return data_x, data_y #list of lists, list

def count_lines(filename):
    """
    Method Docstring Placeholder
    """
    file = open(filename, 'r')
    lines = file.readlines()
    print(len(lines))

def write_feature_matrix_to_file(matrix, labels, write_file):
    ### WILL NOT WORK WITH TWO DIGIT LABELS CARE CARE CARE
    '''
    matrix - list of lists
    labels - list of ints
    write_file - string
    '''
    file = open(write_file, 'a')
    #TODO: raise exceptions instead of asserting
    assert len(matrix) == len(labels), 'len of list of feature matrices != len of list of labels'

    for i, _ in enumerate(matrix):
        matrix[i].append(labels[i])

    for vector in matrix:
        file.write('\n')
        for element in vector:
            file.write(str(element) + ' ')
#write_feature_matrix_to_file([[1,2,3],[4,5,6],[7,8,9]], [67, 68, 69], 'testing69.txt')

def extract_urls(filename):
    '''
    takes a filename.txt with urls separated by spaces.
    '''
    file = open(filename, 'r')
    urls = file.read().split(' ')
    return urls

def prepare_data(file_dict):
    '''
    input : dictionary with string-filename keys, and int - label values
    returns : list of lists (x feature matrix), list of labels (ints)

    -basically returns complete_X and complete_Y
    '''

    feature_matrices = [] #List of feature vectors
    feature_labels = []
    for filename in file_dict:

        #time.sleep(sleep_time)
        xy_data = extract_data(filename, file_dict[filename])
        feature_matrices += xy_data[0]
        feature_labels += xy_data[1]

    return feature_matrices, feature_labels

#training_data = prepare_data(TRAINING_FILE_DICT)
#testing_data = prepare_data(TESTING_FILE_DICT)
#training_data_x = training_data[0]
#training_data_y = training_data[1]
#testing_data_x = testing_data[0]
#testing_data_y = testing_data[1]
#write_feature_matrix_to_file(training_data_x, training_data_y, 'satire_vectors-testing.txt')

def load_data(training_dict, cap=0):
    '''
    training_dict: dictionary of string:int, where string is filename int is label
    cap = max number of data points we want to extract
    '''

    training_data = []
    labels = []
    for file in training_dict:
        #print(os.getcwd())
        with open(file, mode='r') as current:
            data = current.readlines()
            limit = 0
            if cap == 0:
                if 'opinion' in file or 'polarized' in file:
                    limit = len(data) / 2
                else:
                    limit = len(data)
            else:
                if 'opinion' in file or 'polarized' in file:
                    limit = cap / 2
                limit = cap
            #print(limit)

            for i in range(int(limit)):
                if len(data[i]) < 2:
                    continue
                data[i] = data[i].strip().split(' ')
                #TODO: raise exceptions instead of asserting
                assert isinstance(data[i], list), 'not a list bruh'
                data[i].pop(-1) # remove label in the text file
                labels.append(training_dict[file])
                training_data.append(data[i])

    training_data = [[float(i) for i in j] for _, j in enumerate(training_data)]
    return training_data, labels

TRAINING_FILE_DICT = {'real_news_vectors-training.txt' : 1,
                      'fake_news_vectors-training.txt' : 2,
                      'opinion_vectors-training.txt' : 3,
                      'polarized_news_vectors-training.txt' : 5,
                      'satire_vectors-training.txt' : 7}

TESTING_FILE_DICT = {'real_news_vectors-testing.txt' : 1,
                     'fake_news_vectors-testing.txt' : 2,
                     'opinion_vectors-testing.txt' : 3,
                     'polarized_news_vectors-testing.txt' : 5,
                     'satire_vectors-testing.txt' : 7}


def retrieve_data(file_dict, cap):
    '''
    returns:
    X: feature matrix from file dict
    '''

    print('Retreiving data...')
    training_data = load_data(file_dict, cap)
    data_x = training_data[0]
    data_y = training_data[1]
    return data_x, data_y

def svm_classifier(x_feature_matrix, y_labels, kernel='rbf', gamma="scale", random_state=0):
    """
    Method Docstring Placeholder
    """
    support_vector_machine = sklearn.svm.SVC(kernel=kernel, gamma=gamma, random_state=random_state)
    support_vector_machine.fit(x_feature_matrix, y_labels)
    return support_vector_machine

def run_predictions(trained_classifier, test_x):
    '''
    trained_classifier - a trained classifier from sklearn
    test_x - feature matrix for testing the classifier
    test_y - list of labels that correspond to test_x
    '''
    predictions = []
    for vector in test_x:
        predictions.append(trained_classifier.predict(np.array(vector).reshape(1, -1)))
    return predictions

#validate(support_vector_machine, test_x, test_y)
