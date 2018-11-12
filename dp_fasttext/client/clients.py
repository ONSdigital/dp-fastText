"""
Defines the HTTP client for making requests to dp-fasttext
"""
import logging

from numpy import array, ndarray


class SupervisedClient(object):

    def __init__(self, client):
        self.client = client

        self._predict_uri = "/supervised/predict"
        self._sentence_vector_uri = "/supervised/sentence/vector"

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

        json, headers = await self.client.post(uri, data, **kwargs)
        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'get_sentence_vector'", extra={
                "context": headers.get(self.client.REQUEST_ID_HEADER),
                "data": json
            })
            raise Exception("Invalid response for method 'get_sentence_vector'")

        vector = json.get("vector")

        if not isinstance(vector, list) or len(vector) == 0:
            logging.error("Word vector is None/empty", extra={
                "context": headers.get(self.client.REQUEST_ID_HEADER),
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
        json, headers = await self.client.post(uri, data, **kwargs)

        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'predict'", extra={
                "context": headers.get(self.client.REQUEST_ID_HEADER),
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


class UnsupervisedClient(object):

    def __init__(self, client):
        self.client = client

        self._similar_to_vector_uri = "/unsupervised/similar/vector"

    async def similar_by_vector(self, encoded_vector: str, num_labels: int, **kwargs) -> list:
        """
        Queries the unsupervised similar by vector API
        :param encoded_vector:
        :param num_labels:
        :return:
        """
        uri = self._similar_to_vector_uri
        data = {
            "encoded_vector": encoded_vector,
            "num_labels": num_labels
        }
        json, headers = await self.client.post(uri, data, **kwargs)

        if not isinstance(json, dict) or len(json.keys()) == 0:
            logging.error("Invalid response for method 'similar_by_vector'", extra={
                "context": headers.get(self.client.REQUEST_ID_HEADER),
                "query_params": {
                    "encoded_vector": encoded_vector,
                    "num_labels": num_labels
                },
                "data": json
            })
            raise Exception("Invalid response for method 'similar_by_vector'")

        similar_words: list = json.get("words")

        return similar_words
