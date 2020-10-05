![](http://pike.psu.edu/images/sysfake.png?width=600)

# [SysFake](https://sites.google.com/site/pikesysfake/home)
<sup>Terrence Li (Langer81) and Hunter S. DiCicco (dicicch)</sup>

<sup>Under the direction of Dr. Dongwon Lee, Ph.D., Dr. S. Shyam Sundar, Ph.D. and the SysFake team under Penn State's department of Journalism</sup>

<sup>For information about how this research is supported, please visit the homepage linked above.</sup>

## Overview

### Problem Statement
In an online news media environment governed by conflicting interests vying for readers' attention, it is becoming increasingly difficult for the average consumer to tell factual news apart from sensational, fake or otherwise. We imagine a capability (say, a browser extension) that leverages statistical learning based on linguistic and metadata features of a news article in question to provide users with an estimate of whether an article is genuine or not.

### Goal
The goal of this project is to train and validate a multinomial C-Support Vector Machine for the purpose of classifying vectorized news articles under the following categories:

* **1:** Real
* **2:** Fake
* **3:** Opinion
* **5:** Polarized
* **7:** Satire
* **9:** Promotional Content
* **11:** Corrections

In doing so we hope to create a characteristic model consisting of only significant contributions from the most representative features.

For more details on how these labels are defined, see the project's companion paper *[“Fake News” is not simply False Information: A Concept Explication and Taxonomy of Online Content](http://pike.psu.edu/publications/abs19.pdf)*

The resulting model will then be tested against humans in an experiment in which both parties will be asked to classify the same set of new articles.

## Repository Content

### Important modules:

`feature_extraction.py` - `feature_extraction.ArticleVector` is a class that handles vectorization. You can find the definitions of the features (individual elements) in the `fill_vector()` method.

`classifier.py` - Handles SVM initialization.

`validator.py` - Contains useful validation routines.

### Important directories:

`data` - Contains final vectors, some raw data and some intermediate data. This will be reorganized in the near future.

Features from the companion explication paper that have been implemented so far:
1. Reputable URL ending (taken from "reputable_news_sources.txt") | boolean
2. whether or not a URL is from a reputable news source | boolean
3. number of times "Today" is written / total number of words | float
4. number of grammar mistakes | int
5. number of quotations / total number of words | float
6. number of past tense instances / total number of words | float 
7. number of present tense instances / total number of words | float
8. number of times "should" is written / total number of words | float
9. whether or not "opinion" is in the URL | boolean
10. number of words that are in all caps / total number of words | float
11. whether or not a URL is from a satire news source | boolean
12. number of apa errors | int
13. number of proper nouns that occur / total number of words | float
14. number of interjections that occur / total number of words | float
15. number of times "you" occcurs / total number of words | float
16. Whether a URL has a dot gov ending / total number of words | float
17. whether a URL is from an unreputable site (taken from "unreputable_news_sources.txt") | boolean

Important features that have not been implemented:
1. Presence of fact-checking
2. Verification of impartial reporting
3. Narrative conflict
4. Human-centric writing
5. Prominence
6. Written by named, publically known news staff
7. Presence of an *About Us* section
8. Presence of emotionally charged words
9. Metadata
10. Un/verified sources listed

## Current Performance:

### First trial performance on full dataset:

|                 | real | fake | opinion | polarized | satire |
|-----------------|------|------|---------|-----------|--------|
| recall          | 0.70 | 0.96 | 0.03    | 0.30      | 0.90   |
| precision       | 0.83 | 0.88 | 0.50    | 0.25      | 0.51   |
| f1              | 0.76 | 0.91 | 0.06    | 0.27      | 0.65   |
| # misclassified | 68   | 10   | 217     | 156       | 24     |

57.59 percent correct overall, 645 correct out of  1120

### Best generalizable model performance:

Out of 10 trials done with 5-fold randomized-parameter cross-validation.

These results are from the officically supported model, which is a logistic stochastic gradient descent learner applied over the Nyström approximation of the RBF kernel of the taxonomy dataset.

Recall: `tp / (tp + fn)`

Precision: `tp / (tp + fp)`


|             | precision |  recall | f1-score | support
|-------------|-----------|---------|----------|--------
|        real |     0.55  |   0.98  |   0.71   |   319
|        fake |     0.53  |   0.53  |   0.53   |   528
|     opinion |     0.88  |   0.16  |   0.27   |   313
|   polarized |     0.52  |   0.39  |   0.44   |   383
|      satire |     0.49  |   0.79  |   0.61   |   204
| promotional |     0.00  |   0.00  |   0.00   |    22
|                                                       |
|    accuracy |           |         |   0.54   |  1769
|   macro avg |     0.50  |   0.48  |   0.43   |  1769
|weighted avg |     0.58  |   0.54  |   0.50   |  1769

And the associated confusion matrix:

|            | real | fake | opinion | polarized | satire | promotional |
|------------|------|------|---------|-----------|--------|-------------|
|real        |  312 |   4  |    0    |     3     |    0   |     0       |
|fake        |   86 | 282  |    0    |   114     |   46   |     0       |
|opinion     |  132 |  42  |   51    |     9     |   79   |     0       |
|polarized   |   25 | 183  |    5    |   148     |   22   |     0       |
|satire      |   11 |  20  |    2    |     9     |  162   |     0       |
|promotional |    0 |   0  |    0    |     0     |   22   |     0       |

## Usage guide:

In order to use the classifier, first you must collect data. To do this use the prepare_data() method from classifier.py. The input is a dictionary with data text files as keys and their corresponding labels. see training_file_dict as an example. 

1. support_vector_machine = classifier.svm_classifier(train_X_uncombined, train_Y_uncombined)
2. svm_predictions = classifier.run_predictions(support_vector_machine, test_X_uncombined, test_Y_uncombined)
3. get_statistics(test_Y_uncombined, svm_predictions)
4. validate(support_vector_machine, test_X_uncombined, test_Y_uncombined)
^^these lines of code will be how you run the classifier for validation.

***Important note***
data is separated out into urls, vectors, and then split into training and testing. **There is currently no centralized collection of data** (this will be rectified soon). For example "Fake News" data will have 5 files:

1. fake_news_urls-testing.txt - text file with fake news urls separated by spaces for testing 
2. fake_news_urls-training.txt - text file with fake news urls separated by spaces for training
3. fake_news_urls.txt - All fake news URLs compiled into one text file.
4. fake_news_vectors-testing.txt - The corresponding fake news testing URLs, from fake_news_urls-testing but vectorized into their respective features.
5. fake_news_vectors-training.txt - The corresponding fake news training URLs, from fake_news_urls-training but vectorized into their respective features.


## Author Contacts

|        name       |     cell     |          email         |
|-------------------|--------------|------------------------|
| Terence G. Li     | 814-308-4495 | terrencegl10@gmail.com |
| Hunter S. DiCicco | 609-815-5122 |  hsdicicco@gmail.com   |
