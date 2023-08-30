import pandas as pd
import os
import autosklearn.classification
import numpy as np
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import shutil
from os import path
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import copy

# automl ensamble model
def automl_classifier(X_train, y_train,
        time_left_for_this_task=600,
        per_run_time_limit=30,
        tmp_folder="/tmp/autosklearn_resampling",
        folds=5,
        disable_evaluator_output=False, X=None, y=None): 
    print("running automl classifier")

    if path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_left_for_this_task,
        per_run_time_limit=per_run_time_limit,
        tmp_folder=tmp_folder,
        disable_evaluator_output=disable_evaluator_output,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": folds},
    )
    print("fitting automl model")
    automl.fit(X_train, y_train, dataset_name="SGA")

    
    automl.refit(X, y)
    filename = f"uploads/automl.sav"
    pickle.dump(automl, open(filename, 'wb'))

    print("refit automl model")
    automl.refit(X_train.copy(), y_train.copy())

    return automl



# decision tree
def decision_tree_classifier(X_train, y_train,
                            # gridsearch to check which metrics are the best to use
                            min_split = np.array([2, 3, 4, 5, 6, 7]),
                            max_nvl = np.array([3, 4, 5, 6, 7, 9, 11]),
                            alg = ['entropy', 'gini'],
                            cv=10, X=None, y=None):

    print("running decision tree classifier")


    # gridsearch to check which metrics are the best to use
    min_split = min_split
    max_nvl = max_nvl
    alg = alg
    values_grid = {'min_samples_split': min_split, 'max_depth': max_nvl, 'criterion': alg}

    model = DecisionTreeClassifier()

    gridDecisionTree = GridSearchCV(estimator = model, param_grid = values_grid, cv = cv)
    gridDecisionTree.fit(X_train, y_train)

    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.set_params(**gridDecisionTree.best_params_)
    decision_tree.fit(X, y)
    filename = f"uploads/decision_tree.sav"
    pickle.dump(decision_tree, open(filename, 'wb'))

    # running decision tree
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.set_params(**gridDecisionTree.best_params_)
    decision_tree.fit(X_train, y_train)

    return decision_tree


def random_forest_classifier(X_train, y_train, n_estimators = np.array([100]), alg = ['entropy', 'gini'], cv=10, X=None, y=None):
    values_grid = {'n_estimators': n_estimators, 'criterion': alg}

    model = RandomForestClassifier()

    gridRandomForest = GridSearchCV(estimator = model, param_grid = values_grid, cv = cv)
    gridRandomForest.fit(X_train, y_train)

    random_forest = RandomForestClassifier(random_state=0)
    random_forest.set_params(**gridRandomForest.best_params_)
    random_forest.fit(X, y)
    filename = f"uploads/random_forest.sav"
    pickle.dump(random_forest, open(filename, 'wb'))

    # running decision tree
    random_forest = RandomForestClassifier(random_state=0)
    random_forest.set_params(**gridRandomForest.best_params_)
    random_forest.fit(X_train, y_train)

    return random_forest



def svc_classifier(X_train, y_train,
                # gridsearch cv
                    C=np.arange(1,10), cv=5, n_jobs=-1, verbose=2, X=None, y=None):
    print("running support vector classifier (svc)")

    svc_params = {"C": C}

    svc = SVC(kernel = "linear", probability=True)

    svc_cv_model = GridSearchCV(svc, svc_params, 
                                cv = cv, 
                                n_jobs = -1, 
                                verbose = 2 )

    svc_cv_model.fit(X_train, y_train)

    svc_tuned = SVC(probability=True)
    svc_tuned.set_params(**svc_cv_model.best_params_)
    svc_tuned.fit(X, y)
    filename = f"uploads/svc.sav"
    pickle.dump(svc_tuned, open(filename, 'wb'))

    svc_tuned = SVC(probability=True)
    svc_tuned.set_params(**svc_cv_model.best_params_)
    svc_tuned.fit(X_train, y_train)

    return svc_tuned


def knn_classifier(X_train, y_train, k_list=list(range(1,10)), cv=10, X=None, y=None):
    
    knn = KNeighborsClassifier()
    k_values = dict(n_neighbors = k_list)
    grid = GridSearchCV(knn, k_values, cv = cv, scoring = 'accuracy')
    grid.fit(X_train, y_train)

    knn = KNeighborsClassifier()
    knn.set_params(**grid.best_params_)
    knn.fit(X, y)
    filename = f"uploads/knn.sav"
    pickle.dump(knn, open(filename, 'wb'))

    knn = KNeighborsClassifier()
    knn.set_params(**grid.best_params_)
    knn.fit(X_train, y_train)

    return knn


def logistic_regression_classifier(X_train, y_train, X=None, y=None):
    logistic = LogisticRegression(random_state = 1, max_iter=1000)
    logistic.fit(X, y)
    filename = f"uploads/logistic_regression.sav"
    pickle.dump(logistic, open(filename, 'wb'))

    logistic = LogisticRegression(random_state = 1, max_iter=1000)
    logistic.fit(X_train, y_train)

    return logistic

def naive_bayes_classifier(X_train, y_train, X=None, y=None):
    naive_bayes = GaussianNB()
    naive_bayes.fit(X, y)
    filename = f"uploads/naive_bayes.sav"
    pickle.dump(naive_bayes, open(filename, 'wb'))
    
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    return naive_bayes