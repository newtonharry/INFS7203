import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline
from scipy import stats

# manually identify non-gaussian features
non_gaussian_features = [21, 26, 32]


# Create a custom transformer to remove outliers for gaussian features
class GaussianOutlierDetector(BaseEstimator, OutlierMixin):
    def __init__(self, sd=2):
        self.sd = sd

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        likelihoods = np.zeros_like(X, dtype=float)
        filters = np.zeros_like(X, dtype=float)

        for col in range(X.shape[1]):
            # Calculate the mean and standard deviation for the column
            mean = X[:, col].mean()
            std_dev = X[:, col].std()
            # Calculate the PDF (likelihood) for each row in the column
            likelihoods[:, col] = stats.norm.pdf(X[:, col], loc=mean, scale=std_dev)
            filters[:, col] = stats.norm.pdf(
                (mean + (self.sd * std_dev)), loc=mean, scale=std_dev
            )

        condition_mask = np.any(likelihoods < filters, axis=1)

        # Convert the condition mask to -1 values being outliers and 1 values being inliers
        condition_mask = np.where(condition_mask, -1, 1)
        return condition_mask


isolation_forest = IsolationForest()
gaussian_outlier_detector = GaussianOutlierDetector()


# Create a custom transformer to remove outliers for non-gaussian features
def gaussian_outlier_removal(X, y, **kwargs):
    god = GaussianOutlierDetector(**kwargs)
    mask = god.fit_predict(X)
    return X[mask == 1], y[mask == 1]


# Create a custom transformer to remove outliers for non-gaussian features
def isolation_forest_outlier_removal(X, y, **kwargs):
    isolation_forest = IsolationForest(**kwargs)
    mask = isolation_forest.fit_predict(X)
    return X[mask == 1], y[mask == 1]


# Imputers
numerical_imputer = SimpleImputer(strategy="mean")
nominal_imputer = SimpleImputer(strategy="most_frequent")


impute_missing_values = ColumnTransformer(
    [
        ("numerical_imputer", numerical_imputer, list(range(100))),
        ("nominal_imputer", nominal_imputer, list(range(100, 128))),
    ]
)

# Random forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ("impute", impute_missing_values),
        (
            "remove_gaussian_outliers",
            FunctionSampler(func=gaussian_outlier_removal, kw_args={"sd": 3}),
        ),
        (
            "remove_non_gaussian_outliers",
            FunctionSampler(
                func=isolation_forest_outlier_removal,
                kw_args={"contamination": 0.06, "n_jobs": -1},
            ),
        ),
        ("rf", RandomForestClassifier(criterion="entropy", n_jobs=-1)),
    ]
)


def run():
    # Load the training and test data
    data = pd.concat([pd.read_csv("train_1.csv"), pd.read_csv("train_2.csv")])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    test_data = pd.read_csv("test.csv")

    # Train the model on the training data
    rf_trained = rf_pipeline.fit(X, y)

    # Get th F1 score and accuracy of the model on the training data via cross-validation
    cv = cross_validate(
        rf_trained,
        X,
        y,
        cv=10,
        scoring=["f1_macro", "accuracy"],
        return_train_score=True,
        n_jobs=-1,
    )

    # Round the scores to 3 decimal places
    accuracy = round(np.average(cv["test_accuracy"]), 3)
    f1 = round(np.average(cv["test_f1_macro"]), 3)

    # Make a prediction on the test data
    predictions = rf_trained.predict(test_data)
    predictions = predictions.astype(int)

    # Write the predcitions and scores to a csv file
    df_predictions = pd.DataFrame(predictions)
    df_predictions[1] = np.nan
    df_predictions.loc[len(df_predictions)] = [accuracy, f1]

    # Ensure the first 300 rows are of integer type and are rounded to the nearest integer
    # df_predictions.iloc[:300, 0] = df_predictions.iloc[:300, 0].round(0).astype(int)

    df_predictions.to_csv("s4785038.csv", index=False, header=False)

    # Reopen the csv and change the first 300 values from floats to integers
    with open("s4785038.csv", "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for i, line in enumerate(lines):
            if i < 300:
                line = line.split(",")
                line[0] = str(int(float(line[0])))
                line = ",".join(line)
            f.write(line)
        f.truncate()
