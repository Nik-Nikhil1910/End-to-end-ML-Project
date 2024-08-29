import logging
import os
import sys

log_dir = "logs"
log_file_path=os.path.join(log_dir,"running_log.log")
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("MLProjectlogger")
file_handler=logging.FileHandler(log_file_path)
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(module)s : %(message)s')
file_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
stream_handler=logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
stream_handler.setFormatter(formatter)
