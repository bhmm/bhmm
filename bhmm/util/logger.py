import logging
import sys
from bhmm.util import config

def logger(name='BHMM', pattern='%(asctime)s %(levelname)s %(name)s: %(message)s',
           date_format='%H:%M:%S', handler=logging.StreamHandler(sys.stdout)):
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
    _logger = logging.getLogger(name)
    _logger.setLevel(config.log_level())
    if not _logger.handlers:
        formatter = logging.Formatter(pattern, date_format)
        handler.setFormatter(formatter)
        handler.setLevel(config.log_level())
        _logger.addHandler(handler)
        _logger.propagate = False
    return _logger
