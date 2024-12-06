import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_svm_and_predict(train_features, train_target, test_features):
    pipeline = Pipeline([
        ['scaler', StandardScaler()],
        ['clf', SVC(kernel='rbf',
                    C=np.float64(56.898660290183045),
                    class_weight={0: 1.2, 1: 1})]
    ])

    pipeline.fit(train_features[:, [0, 3, 4]], train_target)
    return pipeline.predict(test_features[:, [0, 3, 4]])
