import logging
import sys

log_level = logging.DEBUG


class Logger():
    """
    A class which wraps the python logger. Example usage:
    ::
        import logger
        LOG = logger.Logger(__name__).get()
        LOG.debug("test")
    """
    def __init__(
            self,
            name,
            pattern='%(asctime)s %(levelname)s %(name)s: %(message)s',
            date_format='%H:%M:%S',
            handler=logging.StreamHandler(sys.stdout)
    ):
        """
        Retrieves the logger instance associated to the given name.

        :param name: The name of the logger instance.
        :type name: str
        :param pattern: The associated pattern.
        :type pattern: str
        :param date_format: The date format to be used in the pattern.
        :type date_format: str
        :param handler: The logging handler, by default console output.
        :type handler: FileHandler or StreamHandler or NullHandler

        :return: The logger.
        :rtype: Logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        if not logger.handlers:
            formatter = logging.Formatter(pattern, date_format)
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)
        self._logger = logger

    def get(self):
        """
        Returns the logger instance previously created.

        :return: The logger instance.
        """
        return self._logger