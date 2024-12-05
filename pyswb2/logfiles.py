
import logging

class LogFile:
    def __init__(self, path=None, prefix="log"):
        """Initialize the LogFile object."""
        self.path = path or "."
        self.prefix = prefix
        self.filename = [f"{self.prefix}_0.log", f"{self.prefix}_1.log"]
        self.is_open = [False, False]

        # Set up the logging module
        self.logger = logging.getLogger(self.prefix)
        self.logger.setLevel(logging.DEBUG)

        # File handler setup
        self.file_handler = None

    def open(self, level=logging.INFO):
        """Open a log file and set the logging level."""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        log_file = f"{self.path}/{self.filename[0]}"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        self.is_open[0] = True

        print(f"Log file opened: {log_file}")

    def write(self, message, level=logging.INFO):
        """Write a message to the log file."""
        if not self.is_open[0]:
            raise ValueError("Log file is not open. Please open it first.")

        if level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            raise ValueError("Unsupported log level.")

    def close(self):
        """Close the log file."""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.is_open[0] = False
            print("Log file closed.")


# Example Usage
if __name__ == "__main__":
    log = LogFile(path="../../../../Downloads", prefix="example_log")
    log.open(level=logging.DEBUG)
    log.write("This is an info message.", level=logging.INFO)
    log.write("This is a debug message.", level=logging.DEBUG)
    log.write("This is a warning message.", level=logging.WARNING)
    log.close()
