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

    # np.random.seed(232)
    # zero_target_idx = np.where(train_target == 0)[0]
    # idx = np.hstack([np.random.choice(np.where(train_target == 1)[0], zero_target_idx.size), zero_target_idx])
    # np.random.shuffle(idx)

    pipeline.fit(train_features[:, [0, 3, 4]], train_target)
    return pipeline.predict(test_features[:, [0, 3, 4]])

prefix='public_tests/00_test_data_input/'

X = np.load(prefix + 'train/cX_train.npy')
y = np.load(prefix + 'train/cy_train.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=63)

print(np.unique(y_train, return_counts=True))

y_pred = train_svm_and_predict(X_train, y_train, X_test)
print(accuracy_score(y_test, y_pred))
