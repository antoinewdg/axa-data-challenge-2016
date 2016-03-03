import numpy as np
import sys

from classifier_search import best_classifier
from composite_regressor import CompositeRegressor
from featurize_training_set import load_featurized_training_set
from learn_structure import load_structure
from featurizer import *

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def fill_submission(featurized_filename, submission_filename, out_filename):
    features = load_featurized_training_set(featurized_filename)
    X_train = features.drop(['DATE', 'n_calls'], axis=1).as_matrix().astype(float)
    X_train = StandardScaler().fit_transform(X_train)
    y_train = features.n_calls.as_matrix()

    structure = load_structure()
    submission_df = load_submission(submission_filename)
    submission_features = featurize_all(submission_df, structure['ASS_ASSIGNMENT'])
    X_test = submission_features.drop(['n_calls'], axis=1).as_matrix().astype(float)

    estimator = CompositeRegressor(best_classifier(), SGDRegressor())
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    submission_df.prediction = predicted
    submission_df.to_csv(out_filename, sep='\t', index=False, columns=['DATE', 'ASS_ASSIGNMENT', 'prediction'])


if __name__ == "__main__":
    fill_submission(sys.argv[1], sys.argv[2], sys.argv[3])
