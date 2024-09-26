import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.mlProject.utils.common import read_yaml,create_directories
from src.mlProject.constants import *

class PredictionPipeline:
    def __init__(self):
        #finding the column names
        schema_file_path=SCHEMA_FILE_PATH
        self.schema=read_yaml(schema_file_path)
         # Load column names
        all_cols = self.schema.COLUMNS.keys()  # Use keys to get the column names
        target_col = self.schema.TARGET_COLUMN.name
        
        # Store feature names, excluding the target column
        self.feature_names = [col for col in all_cols if col != target_col]
        
        # Load the trained model
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
        # Load the preprocessing pipeline (log transform and scaler) used during training
        self.preprocessing_pipeline = joblib.load(Path('artifacts/data_transformation/data_preprocessing_pipeline.pkl'))
        
        

    def transform_user_input(self, data):
        """
        Apply the same preprocessing (log transformation and scaling) to user input data
        that was used during model training.
        """
        # Ensure the input data has the correct column names
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=self.feature_names)
        elif isinstance(data, pd.DataFrame):
            data.columns = self.feature_names
        
        # Apply the loaded preprocessing pipeline to the input data
        transformed_data = self.preprocessing_pipeline.transform(data)
        
        return transformed_data
        
    def predict(self, data):
        """
        Process user input data, apply transformations, and make a prediction using the model.
        """
        # Transform user input data
        transformed_data = self.transform_user_input(data)
        
        # Make a prediction using the trained model
        prediction = self.model.predict(transformed_data)

        return prediction
