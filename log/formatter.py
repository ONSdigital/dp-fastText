"""
Add namespace to all log messages
"""
from config import CONFIG
from dp.log.formatters import CustomJsonFormatter


class CustomJsonFormatter(CustomJsonFormatter):
    def __init__(self, *args, **kwargs):
        super(CustomJsonFormatter, self).__init__(
            *args, json_indent=CONFIG.LOGGING.json_logger_indent, coloured_logging=CONFIG.LOGGING.coloured_logging, **kwargs)

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['namespace'] = CONFIG.APP.title
