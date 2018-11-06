"""
Defines the HTTP client for making requests to dp-fasttext
"""
import os
import logging.config

import requests
from requests.models import Response

from numpy import ndarray

from uuid import uuid4

from json import dumps

from urllib import parse as urllib_parse

from dp4py_sanic.config import CONFIG as SANIC_CONFIG
from dp4py_sanic.logging.log_config import log_config as sanic_log_config
logging.config.dictConfig(sanic_log_config)


class Client(object):

    REQUEST_ID_HEADER = "X-Request-Id"

    def __init__(self, host, port):
        self.host = host
        self.port = port

        self._predict_uri = "/supervised/predict"
        self._sentence_vector_uri = "/supervised/sentence/vector"

        if "LOGGING_NAMESPACE" not in os.environ:
            SANIC_CONFIG.LOGGING.namespace = "dp-fasttext-client"

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

    def _post(self, uri: str, data: dict, **kwargs) -> Response:
        """
        Send a POST request to the given uri
        :param data:
        :return:
        """
        target = self.target_for_uri(uri)
        kwargs["headers"] = self.get_headers()

        logging.info("Sending request", extra={
            "context": kwargs["headers"][self.REQUEST_ID_HEADER],
            "params": data,
            "host": self.host,
            "port": self.port,
            "target": uri
        })
        return requests.post(target, data=dumps(data), **kwargs)

    def get_sentence_vector(self, query) -> ndarray:
        """
        Returns the sentence vector for the given query
        :param query:
        :return:
        """
        uri = self._sentence_vector_uri
        data = {
            "query": query
        }
        response: Response = self._post(uri, data)

        json: dict = response.json()
        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'get_sentence_vector'", extra={
                "context": response.headers.get(self.REQUEST_ID_HEADER),
                "data": json
            })
            raise Exception("Invalid response for method 'predict'")

        vector = json.get("vector")

        if not isinstance(vector, list) or len(vector) == 0:
            logging.error("Word vecotr is None/empty", extra={
                "context": response.headers.get(self.REQUEST_ID_HEADER),
                "query_params": {
                    "query": query
                },
                "data": json
            })
            raise Exception("Invalid response for method 'predict'")

        return ndarray(vector)

    def predict(self, query: str, num_labels: int, threshold: float) -> tuple:
        """
        Return model labels for the given query string
        :param query:
        :param num_labels:
        :param threshold:
        :return:
        """
        uri = self._predict_uri
        data = {
            "query": query,
            "num_labels": num_labels,
            "threshold": threshold
        }
        response: Response = self._post(uri, data)

        json: dict = response.json()
        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'predict'", extra={
                "context": response.headers.get(self.REQUEST_ID_HEADER),
                "query_params": {
                    "query": query,
                    "num_labels": num_labels,
                    "threshold": threshold
                },
                "data": json
            })
            raise Exception("Invalid response for method 'predict'")

        labels = json.get("labels")
        probabilities = json.get("probabilities")

        return labels, probabilities
