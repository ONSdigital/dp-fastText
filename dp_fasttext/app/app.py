"""
Code for creating HTTP api app
"""
import asyncio
import uvloop
import logging

from dp_fasttext.config import CONFIG
from dp_fasttext.app.fasttext_server import FasttextServer
from dp_fasttext.api.supervised.routes import supervised_blueprint

from dp4py_sanic.app.exceptions.error_handlers import ErrorHandlers


def create_app() -> FasttextServer:
    """
    Creates the Sanic APP and registers all blueprints
    :return:
    """
    # First, set the ioloop event policy to use uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Now initialise the APP config, logger and ONSRequest handler
    app = FasttextServer(CONFIG.APP.title)

    # Register blueprints
    app.blueprint(supervised_blueprint)

    logging.info("Using config", extra={"config": CONFIG.to_dict()})

    # Register error handlers
    ErrorHandlers.register(app)

    return app
