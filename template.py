import os
import pathlib as Path
import logging
#this allows us to set the basic configuration of logging. Setiing the level to info ,means that only the events of level info and above will be logged i.e.,info,warning,error and critical.
logging.basicConfig(level= logging.INFO, format='[%(asctime)s]: %(message)s')