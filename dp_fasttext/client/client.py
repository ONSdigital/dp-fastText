"""
Defines the HTTP client for making requests to dp-fasttext
"""
import os
import logging.config

import requests
from requests.models import Response

from uuid import uuid4

from json import dumps

from urllib import parse as urllib_parse

from dp_fasttext.config import CONFIG
from dp4py_sanic.config import CONFIG as SANIC_CONFIG
from dp4py_sanic.logging.log_config import log_config as sanic_log_config
logging.config.dictConfig(sanic_log_config)


class Client(object):

    REQUEST_ID_HEADER = "X-Request-Id"

    def __init__(self, host, port):
        self.host = host
        self.port = port

        self.labels_uri = "/supervised/labels"

        if "LOGGING_NAMESPACE" not in os.environ:
            SANIC_CONFIG.LOGGING.namespace = CONFIG.APP.title

    @staticmethod
    def url_encode(params: dict):
        """
        Url encode a dictionary
        :param params:
        :return:
        """
        return urllib_parse.urlencode(params)

    @staticmethod
    def generate_request_id():
        """
        Generates a random uuid request ID
        :param self:
        :return:
        """
        return str(uuid4())

    def get_headers(self):
        """
        Returns headers for requests
        :return:
        """
        return {
            self.REQUEST_ID_HEADER: self.generate_request_id()
        }

    def target_for_uri(self, uri: str) -> str:
        """
        Returns the full url for a given uri
        :param uri:
        :return:
        """
        return "http://{host}:{port}/{uri}".format(
            host=self.host,
            port=self.port,
            uri=uri[1:] if uri.startswith("/") else uri
        )

    def get_labels(self, query: str, num_labels: int, threshold: float) -> Response:
        """
        Return model labels for the given query string
        :param query:
        :param num_labels:
        :param threshold:
        :return:
        """
        headers = self.get_headers()
        data = {
            "query": query,
            "num_labels": num_labels,
            "threshold": threshold
        }

        target = self.target_for_uri(self.labels_uri)

        logging.info("Sending request", extra={
            "context": headers[self.REQUEST_ID_HEADER],
            "params": data,
            "host": self.host,
            "port": self.port,
            "uri": self.labels_uri
        })
        return requests.post(target, headers=headers, data=dumps(data))
