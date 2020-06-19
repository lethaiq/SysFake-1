import random
import pickle
import glob
import os

from collections import Counter

from sklearn.metrics import recall_score, precision_score, f1_score #roc_auc_score
import numpy as np
import classifier

np.set_printoptions(precision=2)

RANDOM_STATE = 4261998
NUM_TRIALS = 30

# combined -> opinion + polarized are same label
TRAINING_FILE_DICT_UNCOMBINED = {'./data/real_news_vectors-training.txt' : 1,'./data/fake_news_vectors-training.txt' : 2,'./data/opinion_vectors-training.txt' : 3,
                    './data/polarized_news_vectors-training.txt' : 5,'./data/satire_vectors-training.txt' : 7}
TRAINING_FILE_DICT_COMBINED = {'./data/real_news_vectors-training.txt' : 1,'./data/fake_news_vectors-training.txt' : 2,'./data/opinion_vectors-training.txt' : 3,
                    './data/polarized_news_vectors-training.txt' : 3,'./data/satire_vectors-training.txt' : 7}
TESTING_FILE_DICT_UNCOMBINED = {'./data/real_news_vectors-testing.txt' : 1,'./data/fake_news_vectors-testing.txt' : 2,'./data/opinion_vectors-testing.txt' : 3,
                    './data/polarized_news_vectors-testing.txt' : 5,'./data/satire_vectors-testing.txt' : 7}
TESTING_FILE_DICT_COMBINED = {'./data/real_news_vectors-testing.txt' : 1,'./data/fake_news_vectors-testing.txt' : 2,'./data/opinion_vectors-testing.txt' : 3,
                    './data/polarized_news_vectors-testing.txt' : 3,'./data/satire_vectors-testing.txt' : 7}

TRAIN_X_COMBINED, TRAIN_Y_COMBINED = classifier.retrieve_data(TRAINING_FILE_DICT_COMBINED, 1000)
TRAIN_X_UNCOMBINED, TRAIN_Y_UNCOMBINED = classifier.retrieve_data(TRAINING_FILE_DICT_UNCOMBINED, 1000)
TEST_X_COMBINED, TEST_Y_COMBINED = classifier.retrieve_data(TESTING_FILE_DICT_COMBINED, 225) 
TEST_X_UNCOMBINED, TEST_Y_UNCOMBINED = classifier.retrieve_data(TESTING_FILE_DICT_UNCOMBINED, 225)

def validate(model, X, Y):
    '''
    model - sklearn model with fit/predict
    X  - feature matrix i.e. list of lists
    Y  - corresponding y values
    '''
    statistics_dict = {}
    predictions = []
    for vector in X:
        predictions.extend(model.predict(np.array(vector).reshape(1, -1)))
    assert len(Y) == len(predictions), 'bruh the predictions and test_Y don\'t match in length'
    total = len(Y)
    #print(predictions)
    correct = 0
    for i, j in enumerate(predictions):
        #print(Y[i] == predictions[i])
        #print(float(predictions[i]))
        #print(Y[i])

        if float(j) == float(Y[i]):
            correct += 1
        else:
            statistics_dict[Y[i]] = statistics_dict.get(Y[i], 0) + 1
    percent_correct = (correct / total) * 100

    print(statistics_dict)
    print(f"This model got {np.round(np.mean(percent_correct), 2)!s}% correct || {correct!s} out of {total!s}.")
    return percent_correct

def get_statistics(true_y, predictions, verbose=0):
    """
    Method docstring placeholder
    """
    #results_dict = {}
    recall = recall_score(true_y, predictions, average = None)
    precision = precision_score(true_y, predictions, average = None)
    f1 = f1_score(true_y, predictions, average = None) #pylint:disable=C0103
    #auc = roc_auc_score(true_y, predictions, average = 'micro')
    nl = '\n'
    if verbose > 0:
        print(f"{'-'*12}{nl}recall:{recall!s}{nl}precision:{precision!s}{nl}f1:{f1!s}{nl}{'-'*12}{nl}")
    #print('auc:', str(auc)) a
    return [recall, precision, f1]

###############################
####Support Vector Machine#####
###############################
#os.chdir("../models")
SUPPORT_VECTOR_MACHINE = classifier.svm_classifier(TRAIN_X_UNCOMBINED,
                                                   TRAIN_Y_UNCOMBINED,
                                                   C=3.0,
                                                   kernel='linear',
                                                   gamma='auto',
                                                   random_state=RANDOM_STATE,
                                                   verbose=False)
#SUBSET_MODELS = glob.glob("./*subset*.pickle")

#with open(max(SUBSET_MODELS, key=os.path.getctime), mode='rb') as filein:
#    SUPPORT_VECTOR_MACHINE = pickle.load(filein)
SVM_PREDICTIONS = classifier.run_predictions(SUPPORT_VECTOR_MACHINE, TEST_X_UNCOMBINED)
#SVM_PREDICTIONS = classifier.run_predictions(SUPPORT_VECTOR_MACHINE, TEST_X_UNCOMBINED, TEST_Y_UNCOMBINED)
STATS = get_statistics(TEST_Y_UNCOMBINED, SVM_PREDICTIONS, verbose=1)
(RECALL, PRECISION, F1) = *STATS,
validate(SUPPORT_VECTOR_MACHINE, TEST_X_UNCOMBINED, TEST_Y_UNCOMBINED)

# SUPPORT_VECTOR_MACHINE_uncombined = classifier.svm_classifier(TRAIN_Y_UNCOMBINED, TRAIN_Y_UNCOMBINED)
# SVM_PREDICTIONS_uncombined = classifier.run_predictions(SUPPORT_VECTOR_MACHINE, TEST_X_UNCOMBINED, TEST_Y_UNCOMBINED)
# get_statistics(TEST_Y_UNCOMBINED, SVM_PREDICTIONS)
# validate(SUPPORT_VECTOR_MACHINE, TEST_X_UNCOMBINED, TEST_Y_UNCOMBINED)

def find_errors(model, vector_data_file, label):
    '''
    returns a dictionary that tells you how many of each category the model incorrectly predicted. 
    IT TELLS NUMBER OF INCORRECT, NOT CORRECT
    '''
    data, labels = classifier.load_data({vector_data_file : label}, cap=225)

    incorrect_predictions = {}
    model_predictions = []

    for vector in data:
        model_predictions.extend(model.predict(np.array(vector).reshape(1, -1)))
    #print(model_predictions[0])
    #print(model_predictions)

    ### the following is a rewrite of previous code from 
    ### commit 30d36b9fb1f7c7ffc18dccc40a106ad426ca4c6a on branch notebooks
    #incorrect_predictions = {labels[i]:incorrect_predictions.get(j, 0) + 1
    #                         for i, j in enumerate(model_predictions)}
    #                         if float(j) != float(labels[i])
    # what does the line above do?
    for i in range(len(model_predictions)):
        if float(model_predictions[i]) != float(label):
            #print(str(model_predictions[i]), str(label))
            incorrect_predictions[model_predictions[i]] = incorrect_predictions.get(model_predictions[i], 0) + 1

    print(incorrect_predictions)
    return incorrect_predictions

#def vector_diagnostics(vector_data_file, label):
#    data, labels = svm_classifier.load_data({vector_data_file : label})

# print('False negatives for Opinion data:')
# find_errors(SUPPORT_VECTOR_MACHINE, 'opinion_vectors-testing.txt', 3)
# print('False negatives for Polarized News data:')
# find_errors(SUPPORT_VECTOR_MACHINE, 'polarized_news_vectors-testing.txt', 5)

for file in TESTING_FILE_DICT_UNCOMBINED:
    title = file[:file.index('_')]
    print(f"False negatives for {title} data ({TESTING_FILE_DICT_UNCOMBINED[file]!s})")
    find_errors(SUPPORT_VECTOR_MACHINE, file, TESTING_FILE_DICT_UNCOMBINED[file])

# serialize and save current model
with open(f"models/model_kernel[{SUPPORT_VECTOR_MACHINE.kernel}]gamma[{SUPPORT_VECTOR_MACHINE.gamma}]rec[{np.round(np.mean(RECALL), 2)}]pre[{np.round(np.mean(PRECISION), 2)}]f1[{np.round(np.mean(F1), 2)}].pickle", mode="wb") as fileout: #pylint:disable=C0301
    pickle.dump(SUPPORT_VECTOR_MACHINE, fileout)