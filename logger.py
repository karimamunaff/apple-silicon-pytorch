import logging
from stat import filemode


def setup_logger(file_name: str):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(filename)s-%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(file_name, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logging.info("START")
