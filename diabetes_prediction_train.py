#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    # parser.add_argument('--max_leaf_nodes', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str)#, default=os.environ['SM_CHANNEL_TRAIN'])

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
    #import pdb; pdb.set_trace()
    data.drop('MemberID', axis = 1, inplace = True)
    data.fillna(0, inplace = True)

    # separate labels from data
    X = data.drop('risk', axis = 1)
    y = data['risk']

    # standardize trg data
    X = StandardScaler().fit_transform(X)

    # split into trg set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Here we support a single hyperparameter, 'max_leaf_nodes'. Note that you can add as many
    # as your training my require in the ArgumentParser above.
    # max_leaf_nodes = args.max_leaf_nodes

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = SVC()
    clf = clf.fit(X_train, y_train)
    
    # print metrics
    trg_accuracy = metrics.accuracy_score(y_train, clf.predict(X_train))
    #trg_average_precision_score = metrics.average_precision_score(y_train, clf.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, clf.predict(X_test))
    #test_average_precision_score = metrics.accuracy_score(y_test, clf.predict(X_test))

    print('trg_accuracy:',trg_accuracy)
    #print('trg_average_precision_score:',trg_average_precision_score)
    print('test_accuracy:',test_accuracy)
    #print('test_average_precision_score:',test_average_precision_score)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf