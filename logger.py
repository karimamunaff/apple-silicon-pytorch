import logging

def get_logger(logger_name:str, file_name:str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger