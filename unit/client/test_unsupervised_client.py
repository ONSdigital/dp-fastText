"""
Tests that the fasttext client properly parses valid/invalid response JSON
"""
from uuid import uuid4
from numpy import array
from unittest import TestCase
from unittest.mock import MagicMock

from unit.utils.async_test import AsyncTestCase

from dp_fasttext.ml.utils import encode_float_list
from dp_fasttext.client.testing.mock_client import MockClient, mock_similar_vector, mock_invalid_response


class UnsupervisedClientTestCase(TestCase, AsyncTestCase):

    def test_similar_vector(self):
        """
        Tests that test_similar_vector correctly parses response JSON
        :return:
        """
        # Build request data
        test_vector = array([1.0, 0.5, 0.0])
        test_vector_encoded = encode_float_list(test_vector)
        num_labels = 5

        data = {
            "encoded_vector": test_vector_encoded,
            "num_labels": num_labels
        }

        # Assert we can call get_sentence_vector cleanly
        # Define the async function to be ran
        async def async_test_function():
            headers = {
                MockClient.REQUEST_ID_HEADER: str(uuid4())
            }

            async def return_fn():
                return mock_similar_vector(), headers

            # Init mock client
            async with MockClient() as client:
                # Mock out _post
                client.post = MagicMock(return_value=return_fn())

                expected_uri = "/unsupervised/similar/vector"

                # Make the call
                words = await client.unsupervised.similar_by_vector(test_vector_encoded, num_labels, headers=headers)

                client.post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)

    def test_similar_vector_invalid(self):
        """
        Tests that test_similar_vector correctly raises an exception for an invalid request
        :return:
        """
        # Build request data
        test_vector = array([1.0, 0.5, 0.0])
        test_vector_encoded = encode_float_list(test_vector)
        num_labels = 5

        data = {
            "encoded_vector": test_vector_encoded,
            "num_labels": num_labels
        }

        # Assert we can call get_sentence_vector cleanly
        # Define the async function to be ran
        async def async_test_function():
            headers = {
                MockClient.REQUEST_ID_HEADER: str(uuid4())
            }

            async def return_fn():
                return mock_invalid_response(), headers

            # Init mock client
            async with MockClient() as client:
                # Mock out _post
                client.post = MagicMock(return_value=return_fn())

                expected_uri = "/unsupervised/similar/vector"

                # Make the call and assert exception raised
                with self.assertRaises(Exception) as context:
                    words = await client.unsupervised.similar_by_vector(test_vector_encoded, num_labels, headers=headers)
                    self.assertIn("Invalid response for method 'similar_by_vector'", str(context))
                client.post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)
