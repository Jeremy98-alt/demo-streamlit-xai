import pandas as pd
import numpy as np
from scipy import stats
import logging
import os
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

class ChurnModel:
    def __init__(self, dataset_path="./streamlit_app/data/churn_dataset.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.preprocessor = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)

    def load_dataset(self):
        self.logger.info("Loading dataset...")
        df_ = pd.read_csv(self.dataset_path, sep=',', on_bad_lines='skip', index_col=False, dtype='unicode')
        self.df = df_.drop(columns=["RowNumber", "CustomerId", "Surname"])

    def get_dataset(self, size=100):
        self.load_dataset()
        self.preprocess_data()
        self.select_subset(size)
        return self.df
    
    def get_categ_features(self):
        return ["Gender", "Geography"]
    
    def get_numerical_features(self):
        return list(set(self.df.columns) - set(["Exited", "Gender", "Geography"]))
    
    def preprocess_data(self):
        self.logger.info("Preprocessing data...")
        # extract categorical and numericals features from the dataset
        categ_lst = self.get_categ_features()
        numerical_cols = self.get_numerical_features()
        self.df[categ_lst] = self.df[categ_lst].astype("string")
        self.df[numerical_cols] = self.df[numerical_cols].astype("float")
        self.df['Exited'] = self.df['Exited'].astype("int")

        # Removing every cell with nan values present and the duplicates
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()

        # Remove outliers
        self.df = self.df[(np.abs(stats.zscore(self.df[numerical_cols])) < 3).all(axis=1)]

    def select_subset(self, size=100):
        self.logger.info("Selecting a sample...")
        # subselect the dataset to be fast in explanation
        self.df = self.df[:size]

    def define_model(self):
        # Define the column transformer
        categ_lst = self.get_categ_features()
        numerical_cols = self.get_numerical_features()

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categ_lst),
                ('num', StandardScaler(), numerical_cols)
            ]
        )
        self.preprocessor = preprocessor

        # Define the pipeline
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])

    def train_model(self):
        self.logger.info("Training the model...")
        X, y = self.df.drop(columns=["Exited"]), self.df["Exited"]
        self.model.fit(X, y)

    def predict(self, data):
        self.logger.info("Predicting the record...")
        return self.model.predict(data)
    
    def get_latest_version(self, model_artifacts_dir):
        # Get a list of all model files in the model_artifacts directory
        model_files = [f for f in os.listdir(model_artifacts_dir) if re.match(r'churn_model_\d+\.pkl', f)]

        # If no model files found, return None
        if not model_files:
            return None

        # Get the latest version
        latest_version = max(model_files)
        return latest_version
    
    def save_model(self):
        # Create the directory if it doesn't exist
        model_artifacts_dir = "./utils/model_artifact/"
        if not os.path.exists(model_artifacts_dir):
            os.makedirs(model_artifacts_dir)

        # Get the latest version
        latest_version = self.get_latest_version(model_artifacts_dir)

        # Increment the version
        if latest_version is not None:
            version_number = int(latest_version.split("_")[-1].split(".")[0]) + 1
        else:
            version_number = 1

        # Construct the new filename
        filename = f"{model_artifacts_dir}churn_model_{version_number}.pkl"

        self.logger.info(f"Saving the model version {version_number}...")
        # Save the model
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_latest_model(self):
        # Define the directory containing model artifacts
        model_artifacts_dir = "./utils/model_artifact/"

        # Get a list of all model files in the model_artifacts directory
        model_files = [f for f in os.listdir(model_artifacts_dir) if re.match(r'churn_model_\d+\.pkl', f)]

        if not model_files:
            print("No model files found.")
            return None

        # Get the latest version
        latest_version = max(model_files)
        latest_model_path = os.path.join(model_artifacts_dir, latest_version)

        # Load the latest model
        with open(latest_model_path, 'rb') as file:
            latest_model = pickle.load(file)

        return latest_model    

if __name__ == "__main__":
    churn_model = ChurnModel()

    # load the dataset
    churn_model.load_dataset()

    # preprocess the dataset
    churn_model.preprocess_data()

    # select a small piece of the dataset
    churn_model.select_subset(size=200)
    
    # define the model parameters
    churn_model.define_model()

    # fit the model
    churn_model.train_model()

    # save the model
    churn_model.save_model()