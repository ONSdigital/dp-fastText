"""
Defines a custom Sanic app with ML model(s) attached
"""
import logging
from dp4py_sanic.app.server import Server

from dp_fasttext.config import CONFIG
from dp_fasttext.ml.supervised import SupervisedModel
from dp_fasttext.api.request.fasttext_request import FasttextRequest
from dp_fasttext.app.ml.supervised_models_cache import get_supervised_model


class FasttextServer(Server):

    def __init__(self, name=None, router=None, error_handler=None,
                 load_env=True,
                 strict_slashes=False,
                 configure_logging=True):
        super(FasttextServer, self).__init__(name=name, router=router, error_handler=error_handler,
                                             load_env=load_env, request_class=FasttextRequest,
                                             strict_slashes=strict_slashes,
                                             configure_logging=configure_logging)

        self.supervised_filename = CONFIG.ML.supervised_model_filename

        # Initialise model
        logging.info("Initialising fastText model", extra={
            "params": {
                "filename": self.supervised_filename
            }
        })
        self.get_supervised_model()
        logging.info("Successfully initialised fastText model", extra={
            "params": {
                "filename": self.supervised_filename
            }
        })

    def get_supervised_model(self) -> SupervisedModel:
        """
        Returns the ONS supervised fasttext model
        :return:
        """
        logging.debug("Fetching cached supervised model", extra={
            "params": {
                "filename": self.supervised_filename
            }
        })

        return get_supervised_model(self.supervised_filename)
