"""
Tests all routes on the /supervised route
"""
from json import dumps
from unit.utils.test_app import FastTextTestApp


class TestSupervisedApi(FastTextTestApp):

    def test_get_sentence_vector(self):
        """
        Tests the /supervised/sentence/vector API
        :return:
        """
        # Set request params
        data = {
            "query": "rpi"
        }

        # Set the target
        target = '/supervised/vector'

        # Assert 200 response
        request, response = self.post(target, 200, data=dumps(data))

        # Check if response JSON is valid
        self.assertTrue(hasattr(response, 'json'), "response should contain JSON")
        json = response.json

        self.assertIsInstance(json, dict, "JSON should be instanceof dict")

        expected_keys = ["query", "vector"]
        for key in expected_keys:
            self.assertIn(key, json, "JSON should contain key '{0}'".format(key))
            self.assertIsNotNone(json.get(key), "value for key '{0}' should not be None")

    def test_get_sentence_vector_bad_request(self):
        """
        Tests the /supervised/sentence/vector API returns a 400 for an invalid request
        :return:
        """
        # Set empty request params
        data = {}

        # Set the target
        target = '/supervised/vector'

        # Assert 200 response
        request, response = self.post(target, 400, data=dumps(data))

    def test_predict(self):
        """
        Tests the /supervised/sentence/vector API
        :return:
        """
        # Set request params
        data = {
            "query": "rpi",
            "num_labels": 5,
            "threshold": 0.0
        }

        # Set the target
        target = '/supervised/predict'

        # Assert 200 response
        request, response = self.post(target, 200, data=dumps(data))

        # Check if response JSON is valid
        self.assertTrue(hasattr(response, 'json'), "response should contain JSON")
        json = response.json

        self.assertIsInstance(json, dict, "JSON should be instanceof dict")

        expected_keys = ["labels", "probabilities"]
        for key in expected_keys:
            self.assertIn(key, json, "JSON should contain key '{0}'".format(key))
            self.assertIsNotNone(json.get(key), "value for key '{0}' should not be None")

    def test_predict_bad_request(self):
        """
        Tests the /supervised/sentence/vector API returns a 400 for an invalid request
        :return:
        """
        # Set empty request params
        data = {}

        # Set the target
        target = '/supervised/predict'

        # Assert 200 response
        request, response = self.post(target, 400, data=dumps(data))
