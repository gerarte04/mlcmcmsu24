import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_svm_and_predict(train_features, train_target, test_features):
    pipeline = Pipeline([
        ['scaler', StandardScaler()],
        ['clf', SVC(kernel='rbf', C=np.float64(56.898660290183045))]
    ])

    pipeline.fit(train_features[:, [0, 3, 4]], train_target)
    return pipeline.predict(test_features[:, [0, 3, 4]])
