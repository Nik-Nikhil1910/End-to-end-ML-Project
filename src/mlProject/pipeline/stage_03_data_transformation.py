from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject import logger

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipeline:
    
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_Transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        transformed_data=data_transformation.remove_skewness()
        x_train, x_test, y_train, y_test=data_transformation.train_test_splitting(transformed_data)
        X_train_scaled, X_test_scaled=data_transformation.scaler(x_train, x_test)
        data_transformation.to_write(X_train_scaled, y_train, X_test_scaled, y_test)
        
 
        
if __name__ == '__main__':
    try:
        with open("artifacts\data_validation\status.txt","r") as f:
            status=f.read().split(" ")[-1]
        if(status=="True"):
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataTransformationTrainingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            raise Exception("your data schema is not valid")
    except Exception as e:
        logger.exception(e)
        raise e