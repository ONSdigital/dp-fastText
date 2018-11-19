"""
Defines custom Request object for this API
"""
from typing import Dict
from numpy import array, ndarray
from sanic.exceptions import InvalidUsage, ServerError

from dp4py_sanic.api.request import Request

from dp_fasttext.log import logger
from dp_fasttext.ml.utils import decode_float_list


class FasttextRequest(Request):

    def get_query_string(self) -> str:
        """
        Returns query string from POST params
        :return:
        """
        if self.json is not None and "query" in self.json:
            return self.json.get("query")

        message = "No query specified"
        logger.error(self.request_id, message)
        raise InvalidUsage(message)

    def get_batch_query_strings(self) -> Dict[str, str]:
        """
        Returns a dictionary of _id to str for batch sentence vector requests
        :return:
        """
        if self.json is not None and "queries" in self.json:
            queries = self.json.get("queries")
            if not isinstance(queries, dict):
                message = "Must supply Dict[id, query]"
                logger.error(self.request_id, message)
                raise InvalidUsage(message)

            return queries
        message = "No query specified"
        logger.error(self.request_id, message)
        raise InvalidUsage(message)

    def get_query_vector(self) -> ndarray:
        """
        Parses input (string encoded) embedding vector and returns np.ndarray
        :return:
        """
        if self.json is not None and "encoded_vector" in self.json:
            encoded_vector: str = self.json.get("encoded_vector")

            # If instance of string, decode
            if isinstance(encoded_vector, str):
                try:
                    encoded_vector: list = decode_float_list(encoded_vector)
                except Exception as e:
                    logger.error(self.request_id, "Caught exception while trying to decode vector", exc_info=e, extra={
                        "params": {
                            "encoded_vector": encoded_vector
                        }
                    })
                    raise ServerError("Caught exception while trying to decode vector")

            return array(encoded_vector)
        raise InvalidUsage("Must supply valid (binary encoded) vector")

    def get_num_labels(self) -> int:
        """
        Returns the number of requested labels from POST params
        :return:
        """
        return self.json.get("num_labels", 10)

    def get_threshold(self) -> float:
        """
        Returns the label threshold from POST params
        :return:
        """
        return self.json.get("threshold", 0.0)
