from src.mlProject.entity.config_entity import ModelTrainerConfig
import joblib
import pandas as pd 
import xgboost as xgb
import os

class ModelTrainer:
    def __init__(self,config) -> None:
        self.config=config
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
        
        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        model = xgb.XGBRegressor(colsample_bytree=self.config.colsample_bytree,learning_rate=self.config.learning_rate,max_depth=self.config.max_depth,n_estimators=self.config.n_estimators,subsample=self.config.subsample,random_state=42)
        
        model.fit(X_train,y_train)
        joblib.dump(model,os.path.join(self.config.root_dir, self.config.model_name))
        