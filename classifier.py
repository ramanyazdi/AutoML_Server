"""
pipeline to train classification models
"""

import classifier_models
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder # not always needed
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn import preprocessing


def prepare_data_helper(df, data_columns, label_column, scale, oversampling):
    # get X and y from the dataset
    y = df[label_column]
    df = df[data_columns]
    
    
#     # finding columns which are non numerical
#     data_column_types = []
#     for d in list(df.dtypes):
# #         print(str(d))
#         if str(d) == 'object':
#             data_column_types.append('Categorical')
#         else:
#             data_column_types.append('Numerical')
        
#     # one hot encoding
#     if 'Categorical' in data_column_types:

#         # extract all the variables which are categorical
#         categorical_variables = []
#         for index, col in enumerate(data_column_types):
#             if col=='Categorical':
#                 categorical_variables.append(data_columns[index])

#                 dummy = pd.get_dummies(df[data_columns[index]])
#                 df = pd.concat([df, dummy], axis=1)
#                 del df[data_columns[index]]


    X = df.values
    
    

    # TODO model output when scaled
    # scaling
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X, y
    

def prepare_data(dataset_path,
                data_columns,
                label_column,
                train_validation_random_split = True,
                train_validation_random_split_random_seed = 0,
                validation_set_size = 0.3,
                scale = False,
                oversampling = False):
    
    # import the dataset
    df = pd.read_csv(dataset_path)
    before =  len(df.index)
    print("dataset has", before, "rows")

    df = df.dropna()
    after = len(df.index)
    print(before-after, "null values removed. New row count is", after)
    print()
    
    # dealing with string columns
    columns = list(df)
    
    for c, dtype in zip(columns, list(df.dtypes)):
        if dtype == 'object':
            le = preprocessing.LabelEncoder()
            le.fit(df[c])
            df[c] = le.transform(df[c])

    # split df into train and validation
    train_df, test_df = train_test_split(df, test_size=validation_set_size, 
                                        random_state=train_validation_random_split_random_seed,
                                        stratify=df[[label_column]])
    # train_df, test_df = train_test_split(df, test_size=validation_set_size, 
    #                                     stratify=df[[label_column]])
    
    test_df.to_csv('results/test_df.csv', index=False)
    
    
    X_train, y_train = prepare_data_helper(train_df, data_columns, label_column, scale, oversampling)
    # oversampling
    # oversampling = True
    if oversampling:
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = ros.fit_resample(X_train, y_train)


    X_cv, y_cv = prepare_data_helper(test_df, data_columns, label_column, scale, oversampling)

    X, y = prepare_data_helper(df, data_columns, label_column, scale, oversampling)

    
    return X_train, X_cv, y_train, y_cv, X, y


def train_models(X_train, y_train, time_left_for_this_task, per_run_time_limit, cv, X, y):
    # code for calling the mdoels one by one
    
    decision_tree = classifier_models.decision_tree_classifier(X_train, y_train, cv=cv, X=X, y=y)
    svc = classifier_models.svc_classifier(X_train, y_train, cv=cv, X=X, y=y)
    automl = classifier_models.automl_classifier(X_train, y_train, time_left_for_this_task, per_run_time_limit, folds=cv, X=X, y=y)
    naive_bayes = classifier_models.naive_bayes_classifier(X_train, y_train, X=X, y=y)
    random_forest = classifier_models.random_forest_classifier(X_train, y_train, cv=cv, X=X, y=y)
    knn = classifier_models.knn_classifier(X_train, y_train, cv=cv, X=X, y=y)
    logistic = classifier_models.logistic_regression_classifier(X_train, y_train, X=X, y=y)

    trained_models = {
                        "automl": automl,
                        "svc":svc,
                        "decision_tree":decision_tree,
                        "random_forest":random_forest,
                        "naive_bayes":naive_bayes,
                        "knn":knn,
                        "logistic_regression":logistic
                    }

    return trained_models

def load_models(path="models"):
    """load models from a directory. Returns models as a dict. key is model name. Value is sklearn model"""
    model_names = os.listdir(path)
    trained_models = {}

    for model in model_names:
        filename = path+'/'+model
        loaded_model = pickle.load(open(filename, 'rb'))
        trained_models[model.split('.')[0]] = loaded_model

    return trained_models


def calculate_permutation(trained_models, X_cv, y_cv, n_repeats=10):
    importances = {}

    for model_name in trained_models:
        clf = trained_models[model_name]
        result = permutation_importance(clf, X_cv, y_cv, n_repeats=n_repeats, random_state=0)
        importances[model_name] = list(result.importances_mean)

    return importances


def save_models(trained_models, path="models"):
    """Takes as input dictionary of models. Key is model name. Value is sklearn model"""
    for model_name in trained_models:
        filename = f"{path}/{model_name}.sav"
        pickle.dump(trained_models[model_name], open(filename, 'wb'))
        print(f"{model_name} saved")


def evaluate_models(trained_models, X_cv, y_cv):
    """
    evaluate model performance
    """
    results = {}
    for key in trained_models:
        predictions = trained_models[key].predict(X_cv)
        np.save(f"results/{key}_predictions_cv.npy", predictions)
        np.save(f"results/{key}_predict_proba_cv.npy", trained_models[key].predict_proba(X_cv))

        # print(f"{key} accuracy score:", accuracy_score(y_cv, predictions))
        # results[key] = accuracy_score(y_cv, predictions)
        # results[key] = balanced_accuracy_score(y_cv, predictions)
        results[key] = {
         'accuracy_score': accuracy_score(y_cv, predictions),
         'balanced_accuracy_score': balanced_accuracy_score(y_cv, predictions)
        }


    return results

    # TODO save results as a file


# TODO if external test set is available
# if external_test_set_available:
#     df = pd.read_csv('datasets/SGA/SGA_testingdata.csv', sep = ',')

#     y_test = df['Outcome']
#     df = df[data_columns]
#     X_test = df.values

#     X_test = scaler.fit_transform(X_test)

#     print()
#     print("Test set results")
#     for key in trained_models:
#         predictions = trained_models[key].predict(X_test)
#         print(f"{key} accuracy score:", accuracy_score(y_test, predictions))





# start inputs to be set
# dataset_path = 'download/diabetes.csv'

# data_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# # TODO find numerical str types automatically converted to Categorical
# data_column_types = len(data_columns)*['Numerical']

# label_column = "Outcome"

# binary_classifier = False
# scale = False
# oversampling = False
# validation_set_size = 0.3

# train_validation_random_split = True
# train_validation_random_split_random_seed = 0

# external_test_set_available = False

# time_left_for_this_task=30
# per_run_time_limit=14
# end inputs to be set


# if name == main

# prepare data
# X_train, X_cv, y_train, y_cv = prepare_data(dataset_path,
#             data_columns,
#             data_column_types,
#             label_column,
#             train_validation_random_split = True,
#             train_validation_random_split_random_seed = 0,
#             validation_set_size = 0.3,
#             scale = True,
#             oversampling = False)


# # train the models
# trained_models = train_models(X_train, y_train, time_left_for_this_task, per_run_time_limit)




# print()
# print("Cross validation set results")
# results = evaluate_models(trained_models, X_cv, y_cv)
# print(results)