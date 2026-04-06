import logging


class Logger():
    def __init__(self, filename, formatter):
        self.logger = logging.getLogger(filename)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.ch)
        self.fh = logging.FileHandler(filename=filename, mode='w')
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

    def debug(self, info):
        self.logger.debug(info)

    def info(self, info):
        self.logger.info(info)

    def warning(self, info):
        self.logger.warning(info)

    def critical(self, info):
        self.logger.critical(info)
