import numpy as np

from sklearn.base import RegressorMixin


from featurize_training_set import load_featurized_training_set


class CompositeRegressor(RegressorMixin):
    def __init__(self, clf, regressor):
        self.clf = clf
        self.regressor = regressor

    def fit(self, X, y):
        classes = np.copy(y)
        pos_idx = classes > 0
        classes[pos_idx] = 1
        self.clf.fit(X, classes)
        self.regressor.fit(X[pos_idx], y[pos_idx])

    def predict(self, X):
        predicted = self.clf.predict(X)
        pos_idx = predicted == 1
        predicted[pos_idx] = self.regressor.predict(X[pos_idx])
        predicted[predicted < 0] = 0
        return predicted


