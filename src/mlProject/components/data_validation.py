import os
from mlProject import logger
import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config=config
    def validate_all_cols(self) -> bool:
        try:
            validation_status = True
            messages = []

            # Read the data
            data = pd.read_csv(self.config.unzip_data_dir)
            column_data_types = data.dtypes.apply(lambda x: str(x)).to_dict()
            all_cols = list(column_data_types.keys())
            data_dtypes = list(column_data_types.values())
            all_schema = list(self.config.all_schema.keys())
            all_schema_dtype = dict(self.config.all_schema)

            # Check for missing columns
            missing_cols = [col for col in all_cols if col not in all_schema]
            if missing_cols:
                validation_status = False
                messages.append(f"Insufficient Keys: Missing columns {missing_cols}")

            # Check for data type mismatches
            mismatched_keys = {key: (column_data_types[key], all_schema_dtype[key]) for key in column_data_types if key in all_schema_dtype and column_data_types[key] != all_schema_dtype[key]}
            if mismatched_keys:
                validation_status = False
                messages.append(f"Type Mismatch: {mismatched_keys}")

            # Write status to file
            with open(self.config.status_file, 'w') as file:
                file.write(f"Validation status is: {validation_status}\n")
                for message in messages:
                    file.write(f"{message}\n")

            return validation_status

        except Exception as e:
            raise e
