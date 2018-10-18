import logging
from dp.config.section import Section
from dp.config.utils import git_sha, bool_env

# APP Config

APP_CONFIG = Section("App config")
APP_CONFIG.title = "dp-fastText"
APP_CONFIG.version = git_sha()

# Logging config

LOGGING_CONFIG = Section("Logging config")
LOGGING_CONFIG.default_level = logging.INFO
LOGGING_CONFIG.coloured_logging = bool_env('COLOURED_LOGGING_ENABLED', False)
LOGGING_CONFIG.pretty_logging = bool_env('PRETTY_LOGGING', False)
LOGGING_CONFIG.json_logger_indent = 4 if LOGGING_CONFIG.pretty_logging else None