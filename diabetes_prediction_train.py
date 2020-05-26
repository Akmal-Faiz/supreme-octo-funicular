from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil
import datetime

import argparse
import csv
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.svm import SVC

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(file, engine="python") for file in input_files ]
    data = pd.concat(raw_data)

    X = data.drop(['risk'], axis = 1)
    y = data['risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())])
    

    preprocessor = ColumnTransformer(
       transformers = [('tx', numeric_transformer, list(X.columns)[1:])],
       remainder = 'drop')
    
    svc = SVC()
    
    final_pipeline = Pipeline(
        steps = [('preprocessor', preprocessor), ('svc', svc)]
    )
    final_pipeline.fit(X_train,y_train)
    
    print(os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(final_pipeline, os.path.join(args.model_dir, "model.joblib"))

def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        
    
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor

def predict_fn(input_data, model):
    """Preprocess input data
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    """
    print(len(input_data))
    pred = model.predict(input_data)
    out = pd.DataFrame(input_data['MemberID'])
    
    out = pd.DataFrame({'prediction':pred}) 
    out['PredictionTypeID'] = 1
    
    now = datetime.datetime.now()
    out['day'] = now.day
    out['month'] = now.month
    out['year'] = now.year
    return out

def output_fn(prediction, accept):
    """Format prediction output
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    print('pred type:',type(prediction))
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), accept, mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), accept, mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))
