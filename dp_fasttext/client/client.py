"""
Defines the HTTP client for making requests to dp-fasttext
"""
import logging.config

import aiohttp

from numpy import array, ndarray

from uuid import uuid4

from json import dumps

from urllib import parse as urllib_parse

from dp4py_sanic.logging.log_config import log_config as sanic_log_config
logging.config.dictConfig(sanic_log_config)


class Client(object):

    REQUEST_ID_HEADER = "X-Request-Id"

    def __init__(self, host, port):
        logging.debug("Initialising aiohttp.ClientSession")

        self.host = host
        self.port = port
        self.session = aiohttp.ClientSession()

        self._health_uri = "/healthcheck"
        self._predict_uri = "/supervised/predict"
        self._sentence_vector_uri = "/supervised/sentence/vector"

    def __enter__(self):
        raise TypeError("Use async with instead")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> 'Client':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """
        Close the underlying aiohttp.ClientSession
        :return:
        """
        logging.debug("Closing aiohttp.ClientSession")
        await self.session.close()
        logging.debug("aiohttp.ClientSession closed successfully")

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

    async def _post(self, uri: str, data: dict, **kwargs) -> tuple:
        """
        Send a POST request to the given uri
        :param data:
        :return:
        """
        target = self.target_for_uri(uri)
        if "headers" not in kwargs:
            kwargs["headers"] = self.get_headers()

        logging.debug("Sending request", extra={
            "context": kwargs["headers"][self.REQUEST_ID_HEADER],
            "params": data,
            "host": self.host,
            "port": self.port,
            "target": uri
        })

        async with self.session.post(target, data=dumps(data), **kwargs) as response:
            headers = response.headers
            json = await response.json()
            return json, headers

    async def _get(self, uri: str, **kwargs):
        """
        Wrapper for GET requests that allows mocking
        :param uri:
        :param kwargs:
        :return:
        """
        target = self.target_for_uri(uri)
        async with self.session.get(target, **kwargs) as response:
            headers = response.headers
            json = await response.json()
            return json, headers

    async def healthcheck(self, headers=None):
        """
        Pings the healthcheck API
        :return:
        """
        json, headers = await self._get(self._health_uri, headers=headers)
        return json, headers

    async def get_sentence_vector(self, query, **kwargs) -> ndarray:
        """
        Returns the sentence vector for the given query
        :param query:
        :return:
        """
        uri = self._sentence_vector_uri
        data = {
            "query": query
        }

        json, headers = await self._post(uri, data, **kwargs)
        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'get_sentence_vector'", extra={
                "context": headers.get(self.REQUEST_ID_HEADER),
                "data": json
            })
            raise Exception("Invalid response for method 'get_sentence_vector'")

        vector = json.get("vector")

        if not isinstance(vector, list) or len(vector) == 0:
            logging.error("Word vector is None/empty", extra={
                "context": headers.get(self.REQUEST_ID_HEADER),
                "query_params": {
                    "query": query
                },
                "data": json
            })
            raise Exception("Invalid response for method 'get_sentence_vector'")

        return array(vector)

    async def predict(self, query: str, num_labels: int, threshold: float, **kwargs) -> tuple:
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
        json, headers = await self._post(uri, data, **kwargs)

        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'predict'", extra={
                "context": headers.get(self.REQUEST_ID_HEADER),
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
