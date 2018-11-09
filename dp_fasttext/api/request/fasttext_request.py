"""
Defines custom Request object for this API
"""
from sanic.exceptions import InvalidUsage

from dp4py_sanic.api.request import Request


class FasttextRequest(Request):

    def get_query_string(self) -> str:
        """
        Returns query string from POST params
        :return:
        """
        if self.json is not None and "query" in self.json:
            return self.json.get("query")

        raise InvalidUsage("No query specified")

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
