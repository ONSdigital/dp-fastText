"""
Defines custom Request object for this API
"""
from dp4py_sanic.api.request import Request


class FasttextRequest(Request):

    def get_query_string(self) -> str:
        """
        Returns query string from POST params
        :return:
        """
        return self.json.get("query")
        # return "rpi"

    def get_num_labels(self) -> int:
        """
        Returns the number of requested labels from POST params
        :return:
        """
        return self.json.get("num_labels")
        # return 10

    def get_threshold(self) -> float:
        """
        Returns the label threshold from POST params
        :return:
        """
        return self.json.get("threshold")
        # return 0.0
