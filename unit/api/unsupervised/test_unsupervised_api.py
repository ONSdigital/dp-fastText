"""
Tests all routes on the /supervised route
"""
from json import dumps
from numpy.random import rand
from unit.utils.test_app import FastTextTestApp

from dp_fasttext.config.config import ML_CONFIG
from dp_fasttext.app.ml.supervised_models_cache import get_supervised_model
from dp_fasttext.ml.supervised.supervised import SupervisedModel

from dp_fasttext.ml.utils import encode_float_list


class TestUnSupervisedApi(FastTextTestApp):

    def test_get_similar_vector(self):
        """
        Tests the /unsupervised/similar/vector API
        :return:
        """
        fname = ML_CONFIG.supervised_model_filename
        model: SupervisedModel = get_supervised_model(fname)

        # Set request params
        vector = rand(model.get_dimension())
        vector_encoded = encode_float_list(vector)

        data = {
            "encoded_vector": vector_encoded,
            "num_labels": 10
        }

        # Set the target
        target = '/unsupervised/similar/vector'

        # Assert 200 response
        request, response = self.post(target, 200, data=dumps(data))

        # Check if response JSON is valid
        self.assertTrue(hasattr(response, 'json'), "response should contain JSON")
        json = response.json

        self.assertIsInstance(json, dict, "JSON should be instanceof dict")

        expected_key = "words"
        self.assertIn(expected_key, json, "JSON should contain key '{0}'".format(expected_key))
        self.assertIsNotNone(json.get(expected_key), "value for key '{0}' should not be None")

    def test_get_similar_vector_bad_request(self):
        """
        Tests the /supervised/sentence/vector API returns a 400 for an invalid request
        :return:
        """
        # Set empty request params
        data = {}

        # Set the target
        target = '/unsupervised/similar/vector'

        # Assert 200 response
        request, response = self.post(target, 400, data=dumps(data))
