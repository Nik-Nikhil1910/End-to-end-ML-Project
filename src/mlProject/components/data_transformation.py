from mlProject import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from src.mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config):
        self.config = config


    def train_test_splitting(self):
        original_data = pd.read_csv(self.config.data_path)
        
        # Copy the original data to avoid modifying it directly
        data = original_data.copy()
        
        # Assuming the last column is the target
        X = data.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]   # Target

        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Log and print shapes
        train_shape = list(x_train.shape)
        train_shape[1] += 1  # Including the target column

        test_shape = list(x_test.shape)
        test_shape[1] += 1  # Including the target column

        logger.info("Split data into training and test sets")
        logger.info(f"Training data shape: {train_shape}")
        logger.info(f"Testing data shape: {test_shape}")

        print(f"Training data shape: {train_shape}")
        print(f"Testing data shape: {test_shape}")

        return x_train, x_test, y_train, y_test


    
    def remove_skewness(self,x_train, x_test, skew_threshold=0.75)->pd.DataFrame:
        """
        Applies log transformation to skewed features in the dataset.
        
        We will read the file here as it will be the first step of our data transformation
        """

        # Identify skewed features
        skewed_features = x_train.skew().index[x_train.skew() > skew_threshold]
        
        
        for feature in skewed_features:
            # Apply log transformation
            x_train[feature] = x_train[feature].apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
            x_test[feature] = x_test[feature].apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
            
        return x_train,x_test
    
    

    def scaler(self, x_train, x_test):
        column_names = x_train.columns
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        X_test_scaled = scaler.transform(x_test)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=column_names)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=column_names)
        return X_train_scaled_df, X_test_scaled_df
    
    
    
    def to_write(self, x_train, y_train, x_test, y_test):
        # Convert inputs to DataFrames
        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)
    
        # Reset indices to avoid issues with concatenation
        x_train_df.reset_index(drop=True, inplace=True)
        y_train_df.reset_index(drop=True, inplace=True)
        x_test_df.reset_index(drop=True, inplace=True)
        y_test_df.reset_index(drop=True, inplace=True)
        print(y_train_df.head())
        # Concatenate features and target columns
        train_transformed = pd.concat([x_train_df, y_train_df], axis=1)
        test_transformed = pd.concat([x_test_df, y_test_df], axis=1)
    
        # Save the transformed datasets to CSV files
        train_transformed.to_csv(os.path.join(self.config.root_dir, "transformed_train_data.csv"), index=False)
        test_transformed.to_csv(os.path.join(self.config.root_dir, "transformed_test_data.csv"), index=False)
    
        logger.info("Data has been transformed and saved to CSV files")
    
    