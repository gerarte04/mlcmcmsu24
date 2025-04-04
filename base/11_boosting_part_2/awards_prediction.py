from catboost import CatBoostRegressor
import pandas as pd
from numpy import ndarray, nan
import json
import os


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    def load_data(filepath):
        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)

    training_data = load_data(train_file)
    testing_data = load_data(test_file)

    dependent_variable = "awards"

    outcome_vector = training_data.pop(dependent_variable)

    for dataset in [training_data, testing_data]:
        if dependent_variable in dataset.columns:
            dataset.drop(dependent_variable, axis=1, inplace=True)

        keys_to_remove = ['keywords']
        for key in keys_to_remove:
            if key in dataset.columns:
                dataset.drop(key, axis=1, inplace=True)

        for col in ['genres', 'directors', 'filming_locations'] + [f'actor_{i}_gender' for i in range(3)]:
            dataset[col] = dataset[col].apply(lambda x: ','.join(x) if isinstance(x, list) else str(x))

        for i in range(3):
            for feat in ['known_movies', 'postogramm', 'age']:
                dataset[f'actor_{i}_{feat}'] = pd.to_numeric(dataset[f'actor_{i}_{feat}'], errors='coerce').fillna(nan)  # Explicit NaN handling

        dataset["actors_mean_age"] = dataset[[f'actor_{i}_age' for i in range(3)]].mean(axis=1, skipna=True)
        dataset["actors_total_postogramm"] = dataset[[f'actor_{i}_postogramm' for i in range(3)]].sum(axis=1, skipna=True)

    model = CatBoostRegressor(
        iterations=350,
        learning_rate=0.1,
        depth=5,
        random_seed=42,
        loss_function='RMSE',
        devices='0:1',
        logging_level='Silent',
        train_dir='/tmp/catboost_info'
    )

    model.fit(training_data, outcome_vector, cat_features=['genres', 'directors', 'filming_locations'] + [f'actor_{i}_gender' for i in range(3)])

    results = model.predict(testing_data)
    results = results.clip(min=0).astype(float)

    return results
