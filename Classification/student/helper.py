# EECS 445 - Winter 2021
# Project 1 - helper.py

import pandas as pd
import numpy as np

import project1
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)


def get_split_binary_data(fname="data/dataset.csv"):
    """
    Reads in the data from fname and returns it using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Also returns the dictionary used to create the feature matrices.
    Input:
        fname: name of the file to be read from.
    """
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    class_size = 2 * positiveDF.shape[0] // 3
    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)


def get_imbalanced_data(dictionary, fname="data/dataset.csv", ratio=0.25):
    """
    Reads in the data from fname and returns imbalanced dataset using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Input:
        dictionary: dictionary to create feature matrix from
        fname: name of the file to be read from.
        ratio: ratio of positive to negative samples
    """
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    negativeDF = negativeDF[: int(ratio * positiveDF.shape[0])]
    positive_class_size = 2 * positiveDF.shape[0] // 3
    negative_class_size = 2 * negativeDF.shape[0] // 3
    X_train = (
        pd.concat([positiveDF[:positive_class_size], negativeDF[:negative_class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[positive_class_size:], negativeDF[negative_class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test)


def get_test_multiclass_training_data(class_size=1500):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)

    """
    for index, row in dataframe.iterrows():
        X.append(row['text'])
        y.append(row['label'])

    from sklearn.feature_extraction.text import TfidfVectorizer
    td = TfidfVectorizer()
    X = td.fit_transform(X).toarray()

    print(X.shape)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                        random_state = 445)

    #classifier = OneVsOneClassifier(MultinomialNB())
    #classifier.fit(X_train, y_train)

    #y_pred = classifier.predict(X_test)
    C_range=[.1, .5, .8, 1, 3, 6, 10]


    # Summarize selected features
    for c in C_range:
        clf = OneVsOneClassifier(LinearSVC(penalty="l2", C=c, dual=True, loss="hinge"))
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(X_train,y_train)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
            y_train, y_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
            clf.fit(X_train, y_train)
            pred_values = clf.predict(X_test)
            acc = project1.performance(y_test, pred_values, "accuracy")
            scores.append(acc)
        scores = np.array(scores).mean()
        print(f'Test Performance on C_value: {c}, accuracy', scores)

    weight_values = {}
    clf = OneVsOneClassifier(LinearSVC(penalty="l2", C=1, dual=True, loss="hinge",
                             random_state=445))
    clf.fit(X_train, y_train)
    pred_values = clf.predict(X_test)
    print(f'testing accuracy', project1.performance(y_test, pred_values, metric='accuracy'))
    #for metric in vecMetric:
    #    print(f'testing {metric}', project1.performance(y_test, y_pred, metric=metric))

    clf = OneVsOneClassifier(SVC(kernel='poly', degree=2, C=419.057575745141, coef0=0.16291905055552514, gamma='auto', random_state=445))
    clf.fit(X_train, y_train)
    pred_values = clf.predict(X_test)
    print("Quadratic SVM with random search and auroc metric: ")
    print("Test Accuracy: ", project1.performance(y_test, pred_values, metric="accuracy"))
    print("Test F1-score: ", project1.performance(y_test, pred_values, metric="accuracy"))
    """
    """
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataframe, random_state=445, test_size=0.30)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1), norm='l2')
    vectorizer.fit(train)
    vectorizer.fit(test)
    x_train = vectorizer.transform(train)
    y_train = train.drop(labels = ['label'], axis=1)
    x_test = vectorizer.transform(test)
    y_test = test.drop(labels = ['label'], axis=1)
    """

    neutralDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    X_train = (
        pd.concat(
            [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
        )
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:], neutralDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )

    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)

def get_multiclass_training_data(class_size=1500):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)

    neutralDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    X_train = (
        pd.concat(
            [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
        )
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project1.extract_dictionary(X_train)
    Y_train = X_train["label"].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)

    return (X_train,Y_train, dictionary)


def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project1.generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will use
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname + ".csv", header=["label"], index=False)
    return
