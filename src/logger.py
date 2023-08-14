import logging
import os
from datetime import datetime

# file 
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# path
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path, exist_ok= True)

# file + path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

#logging setup
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)

# Test component
#if __name__=="__main__":
#    logging.info("Logging started")

