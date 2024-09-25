from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_evaluation import ModelEvaluation
from src.mlProject import logger

STAGE_NAME="Model Evaluation Stage"
class ModelEvaluationTrainingPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_evaluation_obj=ModelEvaluation(config=model_evaluation_config)
        model_evaluation_obj.save_results()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e