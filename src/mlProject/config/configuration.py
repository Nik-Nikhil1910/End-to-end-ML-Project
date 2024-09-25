from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml,create_directories
from src.mlProject.entity.config_entity import DataIngestionConfig
from src.mlProject.entity.config_entity import DataValidationConfig
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.entity.config_entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(self) -> None:
        config_file_path=CONFIG_FILE_PATH
        params_file_path=PARAMS_FILE_PATH
        schema_file_path=SCHEMA_FILE_PATH
        
        self.config=read_yaml(config_file_path)
        self.schema=read_yaml(schema_file_path)
        self.params=read_yaml(params_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def dataingestionconfig(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_Validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema=self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        )

        return data_validation_config
    
    def get_data_Transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
       
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            preprocessing_steps=config.preprocessing_steps
        )

        return data_transformation_config
    
        
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config = self.config.model_trainer
        params=self.params.XGBRegressor
        schema=self.schema.TARGET_COLUMN
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path= config.test_data_path,
            model_name= config.model_name,
            colsample_bytree= params.colsample_bytree,
            learning_rate= params.learning_rate,
            max_depth= params.max_depth,
            n_estimators= params.n_estimators,
            subsample= params.subsample,
            target_column=schema.name
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self)->ModelEvaluationConfig:
        config = self.config.model_evaluation
        params=self.params.XGBRegressor
        schema=self.schema.TARGET_COLUMN
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name
        )
        return model_evaluation_config