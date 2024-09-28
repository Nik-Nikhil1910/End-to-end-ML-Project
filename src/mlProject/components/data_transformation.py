import joblib
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlProject import logger
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.preprocessing_steps=config.preprocessing_steps
    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up
    
    def train_test_splitting(self):
        # Load the data from the path provided in the config
        data = pd.read_csv(self.config.data_path) 
        
        # Assuming the last column is the target
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target

        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
         # Combine the features and target for both train and test sets
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)

        # Create directory if it doesn't exist
        os.makedirs(self.config.root_dir, exist_ok=True)

        # Save the train and test data to CSV files
        train_file_path = os.path.join(self.config.root_dir, "train_data.csv")
        test_file_path = os.path.join(self.config.root_dir, "test_data.csv")

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        logger.info(f"Training data saved to {train_file_path}")
        logger.info(f"Test data saved to {test_file_path}")

        return x_train, x_test, y_train, y_test

    
    def log_transform(self,data):
        return np.log1p(data)
        
    def scaler(self, data):
        """
        Scales the input data using StandardScaler and saves the fitted scaler.

        Parameters:
        - data: DataFrame or array-like object to be scaled.

        Returns:
        - Scaled data after applying standard scaling.
        """
        # Initialize the scaler
        scaler = StandardScaler()

        # Fit and transform the data
        scaled_data = scaler.fit_transform(data)

        # Save the fitted scaler to a file
        scaler_path = os.path.join(self.config.root_dir, "standard_scaler.pkl")
        joblib.dump(scaler, scaler_path)

        logger.info(f"Fitted scaler saved at: {scaler_path}")

        # Create DataFrame for the scaled data
        scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)

        return scaled_data_df
    
    def apply_transforms(self, x_train, x_test, y_train, y_test):
        # Concatenate the training and test sets
        data = pd.concat([x_train, x_test], ignore_index=True)

        # Initialize the intermediate transformation variable
        transformed_data = data  # Start with the original data
        logger.info(f"Preprocessing steps: {self.preprocessing_steps}")

        for step in self.preprocessing_steps:
            if step['name'] == 'log_transform':
                transformed_data = self.log_transform(transformed_data)  # Update with log-transformed data
            if step['name'] == 'scaler':
                transformed_data = self.scaler(transformed_data)  # Update with scaled data

        # After applying all transformations, split back into train and test
        x_train_transformed = transformed_data.iloc[:len(x_train), :]
        x_test_transformed = transformed_data.iloc[len(x_train):, :]

        # Combine the features and target for both train and test sets
        # Ensure the indices of both features and target are reset before concatenation
        x_train_transformed.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        x_test_transformed.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        # Check the shapes of the DataFrames before concatenation
        logger.info(f"x_train_transformed shape: {x_train_transformed.shape}, y_train shape: {y_train.shape}")
        logger.info(f"x_test_transformed shape: {x_test_transformed.shape}, y_test shape: {y_test.shape}")

        # Combine the transformed features and the original target columns
        train_data = pd.concat([x_train_transformed, y_train], axis=1)
        test_data = pd.concat([x_test_transformed, y_test], axis=1)

        # Save the train and test data to CSV files
        train_file_path = os.path.join(self.config.root_dir, "transformed_train_data.csv")
        test_file_path = os.path.join(self.config.root_dir, "transformed_test_data.csv")

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        logger.info(f"Transformed training data saved to {train_file_path}")
        logger.info(f"Transformed test data saved to {test_file_path}")

        return train_data, test_data

    
    
                
    