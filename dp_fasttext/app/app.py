"""
Code for creating HTTP api app
"""
import os
import asyncio
import uvloop
import logging

from dp_fasttext.config import CONFIG

from dp4py_sanic.app.server import Server
from dp4py_sanic.app.exceptions.error_handlers import ErrorHandlers
from dp4py_sanic.config import CONFIG as SANIC_CONFIG


def create_app() -> Server:
    """
    Creates the Sanic APP and registers all blueprints
    :return:
    """
    # First, set the ioloop event policy to use uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Set logging namespace
    if "LOGGING_NAMESPACE" not in os.environ:
        SANIC_CONFIG.LOGGING.namespace = CONFIG.APP.title

    # Now initialise the APP config, logger and ONSRequest handler
    app = Server()

    logging.info("Using config", extra={"config": CONFIG.to_dict()})

    # Register error handlers
    ErrorHandlers.register(app)

    return app
