from dp4py.config.section import Section
from dp_fasttext.config.config import APP_CONFIG, LOGGING_CONFIG

CONFIG = Section("Global App Config")

CONFIG.APP = APP_CONFIG
CONFIG.LOGGING = LOGGING_CONFIG
