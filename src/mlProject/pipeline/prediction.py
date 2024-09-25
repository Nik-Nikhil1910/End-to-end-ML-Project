import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    def transform_user_input(self,data):
        unskewed_data=data.apply(lambda x: np.log1p(x) if x >= 0 else np.nan)
        scaler = StandardScaler()
        scaled_data=scaler.fit_transform(unskewed_data)
        return scaled_data
        
    def predict(self, data):
        transformed_data=self.transform_user_input(self,data)
        prediction = self.model.predict(transformed_data)

        return prediction