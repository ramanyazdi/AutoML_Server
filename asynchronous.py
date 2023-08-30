from celery import Celery
import requests
from flask import jsonify
import classifier
import ast

app = Celery('tasks', broker='amqp://guest:guest@localhost:5672/', backend='db+sqlite:///db.async')
# app = Celery('tasks', broker='redis://localhost:6379', backend='db+sqlite:///db.async')

def status(task_id):
    res = app.AsyncResult(task_id)
    # print(type(res))
    # print("res", res)
    # print("state")
    # print(res.state)
    return jsonify(res.state)


@app.task
def train_model(parameters):
    dataset_filename = parameters['dataset_filename']
    columns = ast.literal_eval(parameters['columns'])

    
    time_left_for_this_task = int(parameters['time_left_for_this_task'])
    per_run_time_limit = int(parameters['per_run_time_limit'])
    label = "Outcome" # TODO collect label


    # prepare data
    X_train, X_cv, y_train, y_cv = classifier.prepare_data("download/"+dataset_filename,
        data_columns=columns,
        data_column_types=len(columns)*['Numerical'], # TODO
        label_column=label,
        train_validation_random_split = True,
        train_validation_random_split_random_seed = 0,
        validation_set_size = 0.2,
        scale = True,
        oversampling = False)

    # train the models
    trained_models = classifier.train_models(X_train, y_train, time_left_for_this_task, per_run_time_limit)


    print()         
    print("Cross validation set results")
    results = classifier.evaluate_models(trained_models, X_cv, y_cv)
    print(results)


    return jsonify(results)


