from .clients import SupervisedClient, UnsupervisedClient

import aiohttp
from uuid import uuid4
from json import dumps

from typing import Tuple, Any
from multidict import CIMultiDictProxy
from urllib import parse as urllib_parse

import logging.config
from dp4py_sanic.logging.log_config import log_config as sanic_log_config

logging.config.dictConfig(sanic_log_config)


class Client(object):
    REQUEST_ID_HEADER = "X-Request-Id"

    def __init__(self, host: str, port: int):
        logging.debug("Initialising aiohttp.ClientSession")

        self.host = host
        self.port = port

        self.session = aiohttp.ClientSession()

        # Add uri for healthcheck
        self._health_uri = "/healthcheck"

        # Attach supervised and unsupervised clients
        self.supervised = SupervisedClient(self)
        self.unsupervised = UnsupervisedClient(self)

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

    async def get(self, uri: str, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        """
        Wrapper for GET requests that allows mocking
        :param uri:
        :param kwargs:
        :return:
        """
        target = self.target_for_uri(uri)
        async with self.session.get(target, **kwargs) as response:
            headers = response.headers
            json: Any = await response.json()
            return json, headers

    async def post(self, uri: str, data: dict, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        """
        Send a POST request to the given uri
        :param uri:
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
            json: Any = await response.json()
            return json, headers

    async def healthcheck(self, headers: dict=None) -> Tuple[Any, CIMultiDictProxy]:
        """
        Pings the healthcheck API
        :return:
        """
        json, headers = await self.get(self._health_uri, headers=headers)
        return json, headers

    @staticmethod
    def url_encode(params: dict) -> str:
        """
        Url encode a dictionary
        :param params:
        :return:
        """
        return urllib_parse.urlencode(params)

    @staticmethod
    def generate_request_id() -> str:
        """
        Generates a random uuid request ID
        :return:
        """
        return str(uuid4())

    def get_headers(self) -> dict:
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
