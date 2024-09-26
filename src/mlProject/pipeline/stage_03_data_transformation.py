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
        # Step 1: Load Configuration
        config = ConfigurationManager()  # Initialize configuration manager
        data_transformation_config = config.get_data_Transformation_config()  # Get data transformation config (ConfigBox type)
        
        # Step 2: Initialize Data Transformation
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Step 3: Split the Data
        x_train, x_test, y_train, y_test = data_transformation.train_test_splitting()
        
        # Step 4: Apply Log Transform to Remove Skewness
        unskewed_x_train, unskewed_x_test = data_transformation.log_transform(x_train, x_test)
        
        # Step 5: Scale Data (Fit and Save Preprocessing Pipeline)
        X_train_scaled, X_test_scaled = data_transformation.scaler(unskewed_x_train, unskewed_x_test)
        
        # Step 6: Save Transformed Data to CSV Files
        data_transformation.to_write(X_train_scaled, y_train, X_test_scaled, y_test)

        # Step 7: Optionally Load and Use the Saved Pipeline for Future Predictions
        self.load_and_use_pipeline(data_transformation_config.root_dir, unskewed_x_test)
    
    def load_and_use_pipeline(self, root_dir, new_data):
        """
        Load the saved data transformation pipeline and apply it to new data.
        """
        pipeline_path = os.path.join(root_dir, "data_preprocessing_pipeline.pkl")  # Use correct pipeline filename
        
        if os.path.exists(pipeline_path):
            logger.info(f"Loading pipeline from {pipeline_path}")
            preprocessing_pipeline = joblib.load(pipeline_path)  # Load the saved pipeline
            # Apply the pipeline to the new data
            new_data_scaled = preprocessing_pipeline.transform(new_data)
            logger.info("Data has been scaled using the loaded pipeline.")
            # Further processing with new_data_scaled (predictions, saving, etc.)
        else:
            logger.error(f"Pipeline file not found at {pipeline_path}")

if __name__ == '__main__':
    try:
        # Step 8: Check if Data Validation Passed
        with open("artifacts/data_validation/status.txt", "r") as f:
            status = f.read().split(" ")[-1]
        
        if status == "True":
            # Start the Data Transformation Stage
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            obj = DataTransformationTrainingPipeline()  # Create pipeline object
            obj.main()  # Execute the main pipeline
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        else:
            raise Exception("Your data schema is not valid")
    
    except Exception as e:
        logger.exception(e)
        raise e
