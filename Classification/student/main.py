"""April 2022.
"""

import pandas as pd
import numpy as np
import itertools
import string
import re


from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from helper import *
import helper

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)

def extract_word(input_string):
    """
    Extracts all words from a input string
    """
    final_list = []
    for punctuation in string.punctuation:
        input_string = input_string.replace(punctuation," ")
    input_string = input_string.lower()
    final_list = input_string.split()
    return final_list

def generate_ngrams(s, n):
    """
    Removed punctuations and creates a list of multiple words based on the value n
    """
    for punctuation in string.punctuation:
        s = s.replace(punctuation," ")
    s = s.lower()

    tokens = [token for token in s.split(" ") if token != ""]

    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def extract_dictionary(df):
    """
    Creates a dictionary of unique words in dataframe
    """
    word_dict = {}
    i = 0
    counter = 0
    while i < len(df.index):
        text = df['text'].iloc[i]
        list = extract_word(text)
        for indWord in list:
            if indWord not in word_dict:
                word_dict[indWord] = counter
                counter = counter + 1
        i = i + 1
    return word_dict

def tfidif_vector(X):
    """
    Uses TFIDFVectorizer to create array dataframe
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    td = TfidfVectorizer()
    X = td.fit_transform(X).toarray()

def n_grams_dictionary(df):
    """
    Creates a dictionary of unique combinations of words in dataframe
    """
    word_dict = {}
    i = 0
    counter = 0
    while i < len(df.index):
        text = df['text'].iloc[i]
        first = generate_ngrams(text, 2)
        third = extract_word(text)
        list = first + third
        for indWord in list:
            if indWord not in word_dict:
                word_dict[indWord] = counter
                counter = counter + 1
        i = i + 1
    return word_dict

def generate_n_grams_matrix(df, word_dict):
    """
    Creates a matrix of unique combinations of words in dataframe of length unique n_grams x numbers of tweets
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    row_number = 0
    for text in df["text"]:
        first = generate_ngrams(text, 2)
        third = extract_word(text)
        list = first + third
        for indWord in list:
            if indWord in word_dict:
                feature_matrix[row_number,word_dict[indWord]]=1
        row_number = row_number + 1
    return feature_matrix


def generate_feature_matrix(df, word_dict):
    """
    Creates a matrix of unique combinations of words in dataframe of length unique words x numbers of tweets
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    row_number = 0
    for text in df["text"]:
        list = extract_word(text)
        for indWord in list:
            if indWord in word_dict:
                feature_matrix[row_number,word_dict[indWord]] += 1
        row_number = row_number + 1
    return feature_matrix


def performance(y_true, Y_pred, metric="accuracy"):
    """
    Get the performance of the model results
    """
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, Y_pred)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, Y_pred, average=None)
    elif metric == 'auroc':
        return metrics.roc_auc_score(y_true, Y_pred)
    elif metric == 'precision':
        return metrics.precision_score(y_true, Y_pred)
    elif metric == 'sensitivity':
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, Y_pred).ravel()
        if fn + tp == 0:
            return 0
        return (tp / (fn + tp))
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, Y_pred).ravel()
        if tn + fp == 0:
            return 0
        return (tn / (tn + fp))
    return 0


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Calculate prediction scores using K Fold cross validation
    """
    skf = StratifiedKFold(n_splits=k)
    skf.get_n_splits(X,y)
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        if metric == "auroc":
            pred_values = clf.decision_function(X_test)
        else:
            pred_values = clf.predict(X_test)
        acc = performance(y_test, pred_values, metric)
        scores.append(acc)

    return np.array(scores).mean()


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """
    Get the best regularization parameter for cross validation and the best cross validation score
    """
    maxCV = 0
    parameter_value = 0
    for c in C_range:
        clf = OneVsOneClassifier(LinearSVC(penalty="l2", C=c, dual=True, loss="hinge"))
        mean_value = cv_performance(clf, X, y, k, metric)
        if mean_value > parameter_value:
            maxCV = c
            parameter_value = mean_value
    print("Metric:", metric)
    print("Best c:", maxCV)
    print("CV Score: ", parameter_value)
    return maxCV


def plot_weight(X, y, penalty, C_range, loss, dual):
    """
    Plot the different regularization parameters results for Support Vector Machines
    """
    norm0 = []
    for c in C_range:
        clf = LinearSVC(penalty=penalty, C=c, dual=dual, loss=loss)
        clf.fit(X, y)
        theta_value = 0
        for coef in clf.coef_[0]:
            if coef != 0:
                theta_value = theta_value + 1
        norm0.append(theta_value)
    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=10, metric="accuracy", param_range=[]):
    """
    Get the optimal results for a Quadratic Support Vector Machines
    """
    best_C_val, best_r_val = 0.0, 0.0
    parameter_value = 0
    for c,r in param_range:
        clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto', random_state=445)
        mean_value = cv_performance(clf, X, y, k, metric)
        if mean_value > parameter_value:
            best_C_val = c
            best_r_val = r
            parameter_value = mean_value
    return best_C_val, best_r_val

def get_average_non_zeroes(X_train):
    """
    Gets the average non zeroes per row
    """
    average_zeroes = 0
    for i in X_train:
        non_zero_counter = 0
        for j in i:
            if j == 1:
                non_zero_counter = non_zero_counter + 1
        average_zeroes = average_zeroes + non_zero_counter
    total = average_zeroes / (X_train.shape[0])
    return total

def get_common_element(X_train, word_dict):
    """
    Get the most common word in the Matrix
    """
    sum = np.sum(X_train, axis=0)
    result = np.where(sum == np.amax(sum, axis=0))
    return list(word_dict.keys())[list(word_dict.values()).index(result[0])]

def one_vs_all_implementation(X_train, Y_train, X_test, Y_test):
    """
    Implementations the One-vs-all strategy which handles multiclass classification issues by training each class against the all the other classes
    """
    Y_train_first = Y_train
    Y_test_first = Y_test
    Y_train_first[Y_train_first == 0] = -1
    Y_test_first[Y_test_first == 0] = -1
    # one vs all implementation
    vecMetric = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    print("One vs all implementation")
    clf = LinearSVC(penalty="l2", C=1, dual=True, loss="hinge")
    clf.fit(X_train, Y_train_first)
    pred_values = clf.predict(X_test)
    for metric in vecMetric:
        if metric != 'auroc':
            print(f'Test Performance on C_value: 1, {metric}', performance(Y_test_first, pred_values, metric=metric))

    Y_train_second = Y_train
    Y_test_second = Y_test
    Y_train_second[Y_train_second == 0] = 1
    Y_test_second[Y_test_second == 0] = 1

    clf = LinearSVC(penalty="l2", C=1, dual=True, loss="hinge")
    clf.fit(X_train, Y_train_second)
    pred_values = clf.predict(X_test)
    for metric in vecMetric:
        if metric != 'auroc':
            print(f'Test Performance on C_value: 1, {metric}', performance(Y_test_second, pred_values, metric=metric))

    Y_train_third = Y_train
    Y_test_third = Y_test

    clf = LinearSVC(penalty="l2", C=1, dual=True, loss="hinge")
    clf.fit(X_train, Y_train_third)
    pred_values = clf.predict(X_test)
    for metric in vecMetric:
        if metric != 'auroc':
            print(f'Test Performance on C_value: 1, {metric}', performance(Y_test_third, pred_values, metric=metric))

def main():
    """
    Program Flow
    """
    
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    # 3a
    print("Question 3A")
    print("The processed sentence is", extract_word('BEST book ever! It\'s great'))

    # 3b
    print("Question 3B")
    print("d: ",len(X_train[0]))
    #3c
    print("Question 3C")
    print("Average number of nonzero features:", get_average_non_zeroes(X_train))
    print("Most common word:", get_common_element(X_train, dictionary_binary))

    # 4.1
    print("Question 4.1A")
    vecMetric = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    for metric in vecMetric:
        select_param_linear(X_train, Y_train, k=5, metric=metric,
                            C_range=[.001, .01, .1, 1, 10, 100, 1000],
                            loss="hinge", penalty="l2", dual=True)
    
    #4c
    print("Question 4.1C")
    clf = LinearSVC(penalty="l2", C=0.01, dual=True, loss="hinge", random_state=445)
    clf.fit(X_train, Y_train)
    pred_auroc = clf.decision_function(X_test)
    pred_values = clf.predict(X_test)
    for  metric in vecMetric:
        if metric == "auroc":
            print(f'{metric}:', performance(Y_test, pred_auroc, metric=metric))
        else:
            print(f'{metric}:', performance(Y_test, pred_values, metric=metric))
    
    # 4d
    print("Question 4.1D")
    plot_weight(X_train, Y_train, penalty="l2", C_range=[.001, .01, .1, 1, 10, 100], loss="hinge", dual=True)

    # 4e
    print("Question 4.1E")
    clf = LinearSVC(penalty="l2", C=0.1, dual=True, loss="hinge", random_state=445)
    clf.fit(X_train, Y_train)
    max_values = np.argpartition(clf.coef_[0], -5)[-5:]
    min_values = np.argpartition(clf.coef_[0], 5)[:5]
    positiveCoef = []
    maxWords = []
    for val in max_values:
        positiveCoef.append(clf.coef_[0][val])
        maxWords.append(list(dictionary_binary.keys())[list(dictionary_binary.values()).index(val)])
        print("coeff: ", clf.coef_[0][val], end=" ")
        print(" word: ", list(dictionary_binary.keys())[list(dictionary_binary.values()).index(val)])
    negativeCoef = []
    minWords = []
    for val in reversed(min_values):
        negativeCoef.append(clf.coef_[0][val])
        minWords.append(list(dictionary_binary.keys())[list(dictionary_binary.values()).index(val)])
        print("coeff: ", clf.coef_[0][val], end=" ")
        print(" word: ", list(dictionary_binary.keys())[list(dictionary_binary.values()).index(val)])

    plt.bar(maxWords,positiveCoef)
    plt.title('Most Positive Words')
    plt.xlabel('Word Name')
    plt.ylabel('Word Coefficient')
    plt.show()

    plt.bar(minWords,negativeCoef)
    plt.title('Most Negative Words')
    plt.xlabel('Word Name')
    plt.ylabel('Word Coefficient')
    plt.show()
    

    # 4.2
    # a
    print("Question 4.2A")
    select_param_linear(X_train, Y_train, k=5, metric="accuracy",
                        C_range=[.001, .01, .1, 1], loss="squared_hinge",
                        penalty="l1", dual=False)

    # b
    print("Question 4.2B")
    plot_weight(X_train, Y_train, penalty="l1", C_range=[.001, .01, .1, 1], loss="squared_hinge", dual=False)

    # 4.3(b)i
    print("Question 4.3B")
    c_params = [.001, .01, .1, 1, 10, 100, 1000]
    r_params = [.001, .01, .1, 1, 10, 100, 1000]
    i = 0
    j = 0
    param_range = []
    while i < 7:
        while j < 7:
            param_range.append([c_params[i], r_params[i]])
            j += 1
        i += 1
    best_c, best_r = select_param_quadratic(X_train, Y_train, k=5, metric="auroc", param_range=param_range)
    clf = SVC(kernel='poly', degree=2, C=best_c, coef0=best_r, gamma='auto', random_state=445)
    clf.fit(X_train, Y_train)
    pred_auroc = clf.decision_function(X_test)
    print("Quadratic SVM with grid search and auroc metric: ")
    print("Best c: ", best_c, end= " ")
    print("Best coeff: ", best_r)
    print("Test Performance: ", performance(Y_test, pred_auroc, metric="auroc"))

    # 4.3 (b)ii
    print("Question 4.3B(ii)")
    param_range = []
    i = 0
    j = 0
    while j < 5:
        np.random.uniform(-3,3,25)
        j += 1
    c_params = np.random.uniform(-3,3,25)
    r_params = np.random.uniform(-3,3,25)
    while i < 25:
        param_range.append([10 ** c_params[i], 10 ** r_params[i]])
        i += 1
    best_c, best_r = select_param_quadratic(X_train, Y_train, k=5, metric="auroc", param_range=param_range)
    clf = SVC(kernel='poly', degree=2, C=best_c, coef0=best_r, gamma='auto', random_state=445)
    clf.fit(X_train, Y_train)
    pred_auroc = clf.decision_function(X_test)
    print("Quadratic SVM with random search and auroc metric: ")
    print("Best c: ", best_c, end= " ")
    print("Best coeff: ", best_r)
    print("Test Performance: ", performance(Y_test, pred_auroc, metric="auroc"))
    

    # 5.1
    # c
    print("Question 5.1: Linear SVM with imbalanced class weights")
    clf = LinearSVC(penalty="l2", C=0.01, loss="hinge", class_weight ={-1: 1, 1: 10}, random_state = 445)
    clf.fit(X_train, Y_train)
    pred_auroc = clf.decision_function(X_test)
    pred_values = clf.predict(X_test)
    for  metric in vecMetric:
        if metric == "auroc":
            print(f'Test Performance on {metric}:', performance(Y_test, pred_auroc, metric=metric))
        else:
            print(f'Test Performance on {metric}:', performance(Y_test, pred_values, metric=metric))
    

    # 5.2
    # a
    print("Question 5.2")
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )
    print("Question 5.2: Linear SVM with imbalanced data set")
    clf = LinearSVC(penalty="l2", C=0.01, loss="hinge", class_weight ={-1: 1, 1: 1}, random_state=445)
    clf.fit(IMB_features, IMB_labels)
    pred_auroc = clf.decision_function(IMB_test_features)
    pred_values = clf.predict(IMB_test_features)
    for  metric in vecMetric:
        if metric == "auroc":
            print(f'Test Performance on {metric}:', performance(IMB_test_labels, pred_auroc, metric=metric))
        else:
            print(f'Test Performance on {metric}:', performance(IMB_test_labels, pred_values, metric=metric))
    

    # 5.3
    
    print("Question 5.3(b): Choosing appropriate class weights")
    weight_values = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    for weight in weight_values:
        clf = LinearSVC(penalty="l2", C=0.01, loss="hinge",
                        class_weight ={-1: weight, 1: 1-weight}, random_state=445)
        scores = cv_performance(clf, IMB_features, IMB_labels, k=5, metric="f1-score")
        print(f'Test Performance on negweight: {weight} pos_weight: {1-weight}', scores)

    clf = LinearSVC(penalty="l2", C=0.01, loss="hinge", class_weight ={-1: 0.7, 1: 0.3})
    clf.fit(IMB_features, IMB_labels)
    pred_auroc = clf.decision_function(IMB_test_features)
    pred_values = clf.predict(IMB_test_features)
    for  metric in vecMetric:
        if metric == "auroc":
            print(f'Test Performance on {metric}:', performance(IMB_test_labels, pred_auroc, metric=metric))
        else:
            print(f'Test Performance on {metric}:', performance(IMB_test_labels, pred_values, metric=metric))
    
    # 5.4
    # roc curve
    print("Question 5.4: ROC Curve")
    clf_even = LinearSVC(penalty="l2", C=0.01, loss="hinge", class_weight ={-1: 1, 1: 1}, random_state=445)
    clf_even.fit(IMB_features, IMB_labels)
    pred_auroc_even = clf_even.decision_function(IMB_test_features)

    clf_uneven = LinearSVC(penalty="l2", C=0.01, loss="hinge", class_weight ={-1: 0.7, 1: 0.3}, random_state=445)
    clf_uneven.fit(IMB_features, IMB_labels)
    pred_auroc_uneven = clf_uneven.decision_function(IMB_test_features)

    fpr1, tpr1, thresh1 = metrics.roc_curve(IMB_test_labels, pred_auroc_even)
    fpr2, tpr2, thresh2 = metrics.roc_curve(IMB_test_labels, pred_auroc_uneven)

    plt.plot(fpr1, tpr1, color='darkorange', label='Evenly weighted classes')
    plt.plot(fpr2, tpr2, color='red', label='Wn = 0.7, Wp = 0.3')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of evenly and unevenly weighted classes')
    plt.legend(loc="lower right")
    plt.show()
    

    # Question 6: Apply a classifier to heldout features, and then use generate_challenge_labels to print the predicted labels
    print("Question 6")
    (
        X_train, Y_train, X_test, Y_test, dictionary
    ) = get_test_multiclass_training_data()

    
    select_param_linear(X_train, Y_train, k=5, metric="accuracy",
                        C_range=[.001, .01, .1, 0.5, 0.8, 1, 1.2, 1.5, 2, 3], loss="hinge",
                        penalty="l2", dual=True)
                        

    clf = OneVsOneClassifier(LinearSVC(penalty="l2", C=0.1, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    clf =OneVsRestClassifier(LinearSVC(penalty="l2", C=1, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    clf =OneVsOneClassifier(LinearSVC(penalty="l2", C=1, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    clf =OneVsRestClassifier(LinearSVC(penalty="l2", C=0.1, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    clf =OneVsOneClassifier(LinearSVC(penalty="l2", C=0.8, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    clf =OneVsRestClassifier(LinearSVC(penalty="l2", C=0.8, loss="hinge", random_state=445))
    clf.fit(X_train, Y_train)
    pred_values = clf.predict(X_test)
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="f1-score"))
    print(f'Test Performance on C_value: ', performance(Y_test, pred_values, metric="accuracy"))

    heldout_features = get_heldout_reviews(dictionary)
    clf =OneVsRestClassifier(LinearSVC(penalty="l2", C=0.8, loss="hinge", random_state=445)).fit(X_train, Y_train)
    # clf.fit(X_train, Y_train)
    pred_values = clf.predict(heldout_features)

    generate_challenge_labels(pred_values, "test")

    print(f'Test Performance on accuracy:', performance(Y_test, pred_values, metric="accuracy"))

    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)
    clf = OneVsRestClassifier(LinearSVC(penalty="l2", C=1, dual=True, loss="hinge"))
    clf.fit(multiclass_features, multiclass_labels)
    pred_values = clf.predict(heldout_features)


    generate_challenge_labels(pred_values, "test")

    C_range=[.001, .01, .1, 1, 10, 100, 1000]

    for c in C_range:
       clf = OneVsOneClassifier(LinearSVC(penalty="l2", C=c, dual=True, loss="hinge", random_state=445))
       clf.fit(X_train, Y_train)
       pred_values = clf.predict(X_test)
       print(f'Test Performance on C_value: {c}, accuracy', performance(Y_test, pred_values, metric="accuracy"))

    heldout_features = get_heldout_reviews(multiclass_dictionary)
    C_range=[.001, .01, .1, 1, 10, 100, 1000]
    for c in C_range:
        clf = LinearSVC(penalty="l2", C=c, dual=True, loss="hinge")
        clf.fit(multiclass_features, multiclass_labels)
        pred_values = clf.predict(heldout_features)
        print(f'Test Performance on C_value: {c} ', performance(Y_test, pred_values, metric="accuracy"))

    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

    clf = OneVsRestClassifier(LinearSVC(penalty="l2", C=0.5, dual=True, loss="hinge", random_state=445))
    clf.fit(multiclass_features, multiclass_labels)
    pred_values = clf.predict(heldout_features)

    generate_challenge_labels(pred_values, "wschmu")



if __name__ == "__main__":
    main()
