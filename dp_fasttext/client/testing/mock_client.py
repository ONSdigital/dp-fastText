"""
Mock fasttext client for unit testing
"""
from typing import Tuple, Any
from multidict import CIMultiDictProxy

from dp_fasttext.client import Client


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
    :param data:
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


class MockClient(Client):

    def __init__(self):
        super(MockClient, self).__init__("test", 1234)

    async def get(self, uri: str, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        raise NotImplementedError("Method 'get' of MockClient must be mocked!")

    async def post(self, uri: str, data: dict, **kwargs) -> Tuple[Any, CIMultiDictProxy]:
        raise NotImplementedError("Method 'post' of MockClient must be mocked!")
