"""
Tests that the fasttext client properly parses valid/invalid response JSON
"""
from uuid import uuid4
from unittest import TestCase
from unittest.mock import MagicMock

from unit.utils.async_test import AsyncTestCase
from dp_fasttext.client.testing.mock_client import MockClient, mock_sentence_vector, mock_labels_api, mock_invalid_response


class ClientTestCase(TestCase, AsyncTestCase):

    def test_get_sentence_vector(self):
        """
        Tests that get_sentence_vector correctly parses response JSON
        :return:
        """
        # Build request data
        query = "rpi"
        data = {
            "query": query
        }

        # Assert we can call get_sentence_vector cleanly
        # Define the async function to be ran
        async def async_test_function():
            headers = {
                MockClient.REQUEST_ID_HEADER: str(uuid4())
            }

            async def return_fn():
                return mock_sentence_vector(data), headers

            # Init mock client
            async with MockClient() as client:
                # Mock out _post
                client._post = MagicMock(return_value=return_fn())

                expected_uri = "/supervised/sentence/vector"

                # Make the call
                vector = await client.get_sentence_vector(query, headers=headers)

                client._post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)

    def test_get_sentence_vector_invalid(self):
        """
        Tests that get_sentence_vector correctly raises an exception for an invalid request
        :return:
        """
        # Build request data
        query = "rpi"
        data = {
            "query": query
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
                client._post = MagicMock(return_value=return_fn())

                expected_uri = "/supervised/sentence/vector"

                # Make the call and assert exception raised
                with self.assertRaises(Exception) as context:
                    vector = await client.get_sentence_vector(query, headers=headers)
                    self.assertIn("Invalid response for method 'get_sentence_vector'", str(context))
                client._post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)

    def test_predict(self):
        """
        Tests that get_sentence_vector correctly parses response JSON
        :return:
        """
        # Build request data
        query = "rpi"
        num_labels = 5
        threshold = 0.0

        data = {
            "query": query,
            "num_labels": num_labels,
            "threshold": threshold
        }

        # Assert we can call get_sentence_vector cleanly
        # Define the async function to be ran
        async def async_test_function():
            headers = {
                MockClient.REQUEST_ID_HEADER: str(uuid4())
            }

            async def return_fn():
                return mock_labels_api(), headers

            # Init mock client
            async with MockClient() as client:
                # Mock out _post
                client._post = MagicMock(return_value=return_fn())

                expected_uri = "/supervised/predict"

                # Make the call
                labels, probabilities = await client.predict(query, num_labels, threshold, headers=headers)

                client._post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)

    def test_predict_invalid(self):
        """
        Tests that get_sentence_vector correctly raises an exception for an invalid request
        :return:
        """
        # Build request data
        query = "rpi"
        num_labels = 5
        threshold = 0.0

        data = {
            "query": query,
            "num_labels": num_labels,
            "threshold": threshold
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
                client._post = MagicMock(return_value=return_fn())

                expected_uri = "/supervised/predict"

                # Make the call and assert exception raised
                with self.assertRaises(Exception) as context:
                    labels, probabilities = await client.predict(query, num_labels, threshold, headers=headers)
                    self.assertIn("Invalid response for method 'get_sentence_vector'", str(context))
                client._post.assert_called_with(expected_uri, data, headers=headers)

        self.run_async(async_test_function)
