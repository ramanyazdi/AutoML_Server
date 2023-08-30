from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import io
import os
import time
import classifier
import ast
from celery.result import AsyncResult
from flask_celery import make_celery
import json
import re
import numpy as np
import requests
from flask import send_from_directory
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = "super secret key"

app.config['UPLOAD_FOLDER'] = 'uploads'

# celery
app.config.update(CELERY_CONFIG={
    'broker_url': 'amqp://guest:guest@localhost:5672/',
    'result_backend': 'db+sqlite:///db.async',
})

celery = make_celery(app)

queue = []


@app.route('/')  
def main():
    return render_template("index.html")


@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(uploads, filename)



@app.route('/reports/<path:path>')
def send_report(path):
    return send_from_directory('reports', path)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        dataset = request.files['dataset']
        dataset.save('downloads/'+dataset.filename)
        print("dataset successfully saved to", 'downloads/'+dataset.filename)

        return jsonify({'dataset saved to':'downloads/'+dataset.filename})



@app.route('/status', methods=['GET'])
def status():
    if request.method == 'GET':
        task_id = request.args['task_id']

        res = celery.AsyncResult(task_id)
        return jsonify(res.state)

@app.route('/result')
def result():
    # Opening JSON file
    with open('results/model_results.json') as json_file:
        data = json.load(json_file)

    return jsonify(data)


@app.route('/permutation', methods=['GET'])
def permutation():
    trained_models = classifier.load_models()

    y_cv = np.load('results/y_cv.npy')
    X_cv = np.load('results/X_cv.npy')

    permutation_list = classifier.calculate_permutation(trained_models, X_cv, y_cv)

    # permutation_list = {
    #     "automl": [
    #         -0.003463203463203468,
    #         0.09826839826839824,
    #         0.0073593073593073545,
    #         0.00043290043290040715,
    #         -0.006493506493506518,
    #         0.024675324675324684,
    #         0.007792207792207806,
    #         0.010389610389610383
    #     ],
    #     "decision_tree": [
    #         0.0,
    #         0.08917748917748922,
    #         0.0,
    #         0.0,
    #         0.0034632034632035014,
    #         0.009090909090909104,
    #         0.0,
    #         0.0
    #     ],
    #     "svc": [
    #         -0.0030303030303030385,
    #         0.0844155844155844,
    #         -0.004761904761904745,
    #         -0.022510822510822537,
    #         0.005627705627705637,
    #         0.0008658008658008698,
    #         -0.006493506493506496,
    #         0.0038961038961039087
    #     ]
    # }
    return jsonify(permutation_list)


@app.route('/predictions', methods=['GET'])
def getPredictions():
    if request.method == 'GET':
        y_cv = np.load('results/y_cv.npy').tolist()

        # load model_results json file
        with open('results/model_results.json') as json_file:
            data = json.load(json_file)
        
        predictions_dict = {}
        predictions_int_dict = {}
        for key in data:
            # predict proba
            predictions = np.load(f"results/{key}_predict_proba_cv.npy").tolist()
            predictions_dict[key] = predictions

            # predict integer
            predictions = np.load(f"results/{key}_predictions_cv.npy").tolist()
            predictions_int_dict[key] = predictions


        return jsonify({'y_cv':y_cv, 'predictions':predictions_dict, 'predictions_int_dict':predictions_int_dict})


@app.route('/results_csv', methods=['GET'])
def create_results_csv():
    test_df = pd.read_csv('results/test_df.csv')
    files = os.listdir('results')

    files = [f for f in files if f.endswith('_predict_proba_cv.npy')]

    for f in files:
        y_cv = np.load('results/'+f)
        model_name = f[:-21]

        for feature in range(y_cv.shape[1]):
            print(model_name, 'results/'+f, feature)
            print("dataframe_length", len(test_df))
            print("y_cv length", len(y_cv[:, feature]))
            print()

            test_df['prob_class_'+str(feature)] = y_cv[:, feature]
        
        test_df.to_csv('uploads/'+model_name+'_predict_proba_cv.csv', index=False)


    return jsonify({"files":files})



@app.route('/train_model', methods=['POST'])
def index():
    """
    call the model training code
    """
    if request.method == 'POST':
        parameters = request.args
        print("parameters recieved by linux server", parameters)

        # uncomment when testing done
        result = train_models2(parameters)
        # result = {"automl":2,"svm":3,"sklearn":5}

        return jsonify(result)

        train_job = train_models.delay(parameters)

        return jsonify(train_job.id)


def train_models2(parameters):
    print("type of parameters", type(parameters))
    parameters = parameters.to_dict(flat=False)
    print("PARAMETERS", parameters)

    dataset_filename = parameters['dataset_filename'][0]


    columns = re.sub('[^A-Za-z0-9,_-]+', '', parameters['columns'][0]).split(',')
    print()
    print("COLUMNS", columns)
    print()

    # columns = ast.literal_eval(parameters['columns'][0])
    
    time_left_for_this_task = int(parameters['time_left_for_this_task'][0])
    per_run_time_limit = int(parameters['per_run_time_limit'][0])
    cv = int(parameters['cv'][0])
    label = parameters['label'][0] # "Outcome" #
    test_set_size = int(parameters['test_set_size'][0])
    test_set_size = test_set_size/100 # percentage to decimal

    # prepare data
    X_train, X_cv, y_train, y_cv, X, y = classifier.prepare_data("/mnt/d/Dataset2/"+dataset_filename,
        data_columns=columns,
        label_column=label,
        train_validation_random_split = True,
        train_validation_random_split_random_seed = 0,
        validation_set_size = test_set_size,
        scale = True,
        oversampling = False)

    # scaling and mean normalisation
    # scaler = StandardScaler()
    # scaler.fit(X)

    # X = scaler.transform(X)
    # X_train = scaler.transform(X_train)
    # X_cv = scaler.transform(X_cv)

    # save validation data
    np.save('results/X_cv.npy', X_cv)
    np.save('results/y_cv.npy', y_cv)

    # train the models
    trained_models = classifier.train_models(X_train, y_train, time_left_for_this_task, per_run_time_limit, cv, X, y)

    # save the models
    classifier.save_models(trained_models)

    print()         
    print("Cross validation set results")
    results = classifier.evaluate_models(trained_models, X_cv, y_cv)
    print(results)

    
    with open("results/model_results.json", "w") as outfile:
        json.dump(results, outfile)
        print("result has been saved")

    return results




if __name__ == '__main__':
    app.run(debug = True)