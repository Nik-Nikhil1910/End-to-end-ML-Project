from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml,create_directories
from src.mlProject.entity.config_entity import DataIngestionConfig
from src.mlProject.entity.config_entity import DataValidationConfig

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