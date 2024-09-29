import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.mlProject.utils.common import read_yaml, create_directories , load_json
from src.mlProject.constants import *
import logging

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        # Finding the column names
        schema_file_path = SCHEMA_FILE_PATH
        self.schema = read_yaml(schema_file_path)
        
        # Load column names
        all_cols = self.schema.COLUMNS.keys()  # Use keys to get the column names
        target_col = self.schema.TARGET_COLUMN.name
        
        # Store feature names, excluding the target column
        self.feature_names = [col for col in all_cols if col != target_col]
        
        # Load the trained model
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessing_steps=load_json(Path("artifacts/data_transformation/model_metadata.json"))
        
        # Load the preprocessing pipeline (log transform and scaler) used during training
        self.preprocessing_pipeline = joblib.load(Path('artifacts/data_transformation/standard_scaler.pkl'))

    def transform_user_input(self, data):
        """
        Apply the same preprocessing (log transformation and scaling) to user input data
        that was used during model training.
        """
        # Ensure the input data has the correct column names and is a DataFrame
        # Ensure the input data has the correct column names and is a DataFrame
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)
        elif isinstance(data, pd.DataFrame):
            if set(data.columns) != set(self.feature_names):
                logger.error("Input DataFrame columns do not match expected columns: %s", self.feature_names)
                raise ValueError(f"Input DataFrame columns do not match expected columns: {self.feature_names}")
            if set(data.columns) != set(self.feature_names):
                logger.error("Input DataFrame columns do not match expected columns: %s", self.feature_names)
                raise ValueError(f"Input DataFrame columns do not match expected columns: {self.feature_names}")
            data.columns = self.feature_names

        # Debugging: Log input data
        logger.info("Input Data Preview:")
        logger.info("\n%s", data.head())
        
        step_list = self.preprocessing_steps["preprocessing_steps"]
        transformed_data=data
        # Apply the loaded preprocessing pipeline to the input data
        for step in step_list:
            if step["name"]=="log_transform":
                transformed_data=np.log1p(transformed_data)
            elif step["name"]=="scaler":
                transformed_data = self.preprocessing_pipeline.transform(data)

        # Convert transformed data back to DataFrame for consistency
        transformed_data_df = pd.DataFrame(transformed_data, columns=self.feature_names)

        # Debugging: Log transformed data
        logger.info("Transformed Data Preview:")
        logger.info("\n%s", transformed_data_df.head())
    
        # Write the DataFrame to a CSV file
        file_path = Path("artifacts/data_transformation/transformed_prediction_data.csv")
        transformed_data_df.to_csv(file_path, index=False)
    
        return transformed_data_df
    
    def predict(self, data):
        """
        Process user input data, apply transformations, and make a prediction using the model.
        """
        # Transform user input data
        transformed_data = self.transform_user_input(data)
        
        # Make a prediction using the trained model
        prediction = self.model.predict(transformed_data)

        return prediction