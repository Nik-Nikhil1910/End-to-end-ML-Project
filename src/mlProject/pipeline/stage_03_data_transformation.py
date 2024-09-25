from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject import logger
import joblib
import os

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_Transformation_config()
        
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Split the data
        x_train, x_test, y_train, y_test = data_transformation.train_test_splitting()
        
        # Remove skewness
        unskewed_x_train, unskewed_x_test = data_transformation.remove_skewness(x_train, x_test)
        
        # Fit and save the preprocessing pipeline
        X_train_scaled, X_test_scaled = data_transformation.scaler(unskewed_x_train, unskewed_x_test)
        
        # Write transformed data to CSV
        data_transformation.to_write(X_train_scaled, y_train, X_test_scaled, y_test)

        # Optionally: Load and use the saved preprocessing pipeline for predictions
        self.load_and_use_pipeline(data_transformation_config.root_dir, unskewed_x_test)
    
    def load_and_use_pipeline(self, root_dir, new_data):
        """
        Load the saved data transformation pipeline and apply it to new data.
        """
        pipeline_path = os.path.join(root_dir, "data_transformation_pipeline.pkl")
        
        if os.path.exists(pipeline_path):
            logger.info(f"Loading pipeline from {pipeline_path}")
            preprocessing_pipeline = joblib.load(pipeline_path)
            # Apply the pipeline to the new data
            new_data_scaled = preprocessing_pipeline.transform(new_data)
            logger.info("Data has been scaled using the loaded pipeline.")
            # Do something with new_data_scaled (e.g., predictions, saving, etc.)
        else:
            logger.error(f"Pipeline file not found at {pipeline_path}")

if __name__ == '__main__':
    try:
        with open("artifacts/data_validation/status.txt", "r") as f:
            status = f.read().split(" ")[-1]
        
        if status == "True":
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataTransformationTrainingPipeline()
            obj.main()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            raise Exception("Your data schema is not valid")
    except Exception as e:
        logger.exception(e)
        raise e
