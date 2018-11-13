"""
Mock fasttext client for unit testing
"""
from typing import Tuple, Any
from multidict import CIMultiDictProxy

from unittest.mock import MagicMock

from dp_fasttext.client import Client


class MockClient(Client):

    def __init__(self):
        super(MockClient, self).__init__("test", 1234)

    async def get(self, uri: str, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        raise NotImplementedError("Method 'get' of MockClient must be mocked!")

    async def post(self, uri: str, data: dict, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        raise NotImplementedError("Method 'post' of MockClient must be mocked!")


def mock_labels_api() -> dict:
    """
    Returns mock labels and their probabilities
    :return:
    """
    labels = ['economy', 'inflation']
    probabilities = [0.8, 0.4]

    return {
        "labels": labels,
        "probabilities": probabilities
    }


def mock_sentence_vector(data: dict) -> dict:
    """
    Returns a mock sentence vector
    :return:
    """
    vector = [1.0, 0.5, 0.0]
    return {
        "query": data.get("query"),
        "vector": vector
    }


def mock_similar_vector() -> dict:
    """
    Returns a mock list of labels for the input vector
    :return:
    """
    return {
        "words": ['economy', 'inflation']
    }


def mock_invalid_response():
    """
    Returns invalid mock labels and their probabilities
    :return:
    """
    return "Internal server error"


async def empty_get(*args, **kwargs):
    """
    Defines a noop get method
    :return:
    """
    return {}, {}


async def mock_get(uri: str, **kwargs):
    """
    Mocks the get method for health checks
    :param uri:
    :param kwargs:
    :return:
    """
    headers = kwargs.get("headers", {})

    if uri == "/healthcheck":
        return {}, headers
    else:
        raise NotImplementedError("Mock get not implemented for uri '{0}'".format(uri))


async def mock_post(uri: str, data: dict, **kwargs):
    """
    Mocks behaviour of post request
    :param self:
    :param uri:
    :param data:
    :param kwargs:
    :return:
    """
    headers = kwargs.get("headers", {})

    if uri == "/supervised/predict":
        return mock_labels_api(), headers
    elif uri == "/supervised/sentence/vector":
        return mock_sentence_vector(data), headers
    elif uri == "/unsupervised/similar/vector":
        return mock_similar_vector(), headers
    else:
        raise NotImplementedError("Mock post not implemented for uri '{0}'".format(uri))


def mock_fasttext_client() -> MockClient:
    """
    Returns a mocked fasttext client
    :return:
    """
    # Initialise the MockClient and mock the 'get' and 'post' methods
    client = MockClient()
    client.get = MagicMock()
    client.post = MagicMock()

    # Set side effect to new method so we can preserve calling arguments
    client.get.side_effect = mock_get
    client.post.side_effect = mock_post

    return client
