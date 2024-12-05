import logging
from pathlib import Path
from typing import Optional

class ParameterLogger:
    def __init__(self, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger('parameters')
        self.logger.setLevel(logging.INFO)

        if log_dir:
            file_handler = logging.FileHandler(log_dir / 'parameters.log')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)
