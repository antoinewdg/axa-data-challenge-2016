import scipy.stats as st
import sys

from numpy.core.multiarray import dtype
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from learn_structure import load_structure
from featurize_training_set import load_featurized_training_set


def search_classifier(n_iter):
    assignments = load_structure()['ASS_ASSIGNMENT']
    features = load_featurized_training_set("files/train_featurized.pkl")

    # print(len(features.columns))
    X = features.drop(['DATE', 'n_calls'], axis=1).as_matrix().astype(float)
    y = (features.n_calls > 0).astype(int).as_matrix()
    calls = features.n_calls.as_matrix()

    X = StandardScaler().fit_transform(X)
    pipe = Pipeline([
        # ('scaler', StandardScaler()),
        # ('pca', RandomizedPCA()),
        ('clf', SGDClassifier())
    ])

    params = {
        # 'pca__n_components': [30, 50, 70, 86],
        'clf__class_weight': ['balanced'],
        'clf__loss': ['hinge'],
        'clf__penalty': ['l1'],
        'clf__alpha': st.uniform(0, 0.0003),
        'clf__fit_intercept': [False]
        # 'clf__alpha': [0.0001]
    }

    kf = KFold(len(X), n_folds=3, shuffle=True)
    grid_search = RandomizedSearchCV(pipe, params, scoring='accuracy', cv=kf, verbose=1000, n_iter=n_iter)
    grid_search.fit(X, y)

    print("\n")
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    joblib.dump(grid_search.best_estimator_, "files/best_classifier.pkl")


def best_classifier():
    return joblib.load("files/best_classifier.pkl")


if __name__ == "__main__":
    search_classifier(int(sys.argv[1]))
