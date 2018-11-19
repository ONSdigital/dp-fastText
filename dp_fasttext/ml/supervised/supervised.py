"""
This file defines classes and methods for working with supervised fastText models
"""
import logging
import fastText
from typing import Dict

import numpy as np

from dp_fasttext.ml.utils import cosine_similarity, encode_float_list


class SupervisedModel(fastText.FastText._FastText):
    def __init__(self, filename: str, label_prefix: str="__label__"):
        super(SupervisedModel, self).__init__(model=filename)
        self.filename = filename

        logging.info("Initialised supervised fastTet model", extra={
            "model": {
                "filename": filename,
                "prefix": label_prefix,
                "matrix": "({num_words}, {dimensions})".format(num_words=len(self.get_words()), dimensions=self.get_dimension())
            }
        })

        self.label_prefix = label_prefix

    @staticmethod
    def _normalise_matrix(matrix) -> np.ndarray:
        """
        Normalise a matrix of word vectors
        :param matrix:
        :return:
        """
        norm_vector = np.linalg.norm(matrix, axis=1)
        normed_matrix = np.zeros(matrix.shape)

        for i in range(len(matrix)):
            normed_matrix[i] = matrix[i] / norm_vector[i]
        return normed_matrix

    def batch_get_sentence_vector(self, id_text_map: Dict[str, str]):
        """
        Given a dict of id to string, get a single vector represenation for each entry.
        """
        # Build dict to hold results
        results = {}

        # Iterate over id map
        for _id in id_text_map:
            text = id_text_map.get(_id)
            vector: np.ndarray = self.get_sentence_vector(text)

            # Encode
            encoded_vector: str = encode_float_list(list(vector.tolist()))

            # Generate keywords
            keywords, probabilities = self.predict(text, k=10, threshold=0.0)

            keyword_prob_map = {}
            for k, p in zip(keywords, probabilities):
                keyword_prob_map[k] = p

            # Add to results
            results[_id] = {
                "vector": encoded_vector,
                "keywords": keyword_prob_map
            }

        return results

    def predict(self, text, k=1, threshold=0.0):
        """
        Predict labels given raw text.
        :param text: The raw text to predict keyword labels
        :param k: Number of predictions to make
        :param threshold: Minimum likelihood probability
        :return:
        """
        labels, probabilities = super(
            SupervisedModel, self).predict(
            text, k=k, threshold=threshold)

        parsed_labels = []
        for label in labels:
            parsed_labels.append(label.replace(self.label_prefix, ""))

        return parsed_labels, probabilities

    def keywords(self, text, top_n=10) -> list:
        """
        Return predicted keywords, sorted by similarity, for the input text segment
        :param text:
        :param top_n:
        :return:
        """
        labels, proba = self.predict(text, k=top_n)

        result = [{"label": label, "P": P} for label, P in zip(labels, proba)]

        # Sort by probability
        result = sorted(result, key=lambda item: item["P"], reverse=True)
        return result

    def similarity_by_word(self, word1: str, word2: str) -> float:
        """
        Computes the similarity between two words, as measured by their cosine distance
        :param word1:
        :param word2:
        :return:
        """
        vec1: np.ndarray = self.get_word_vector(word1)
        vec2: np.ndarray = self.get_word_vector(word2)
        return self.similarity_by_vector(vec1, vec2)

    @staticmethod
    def similarity_by_vector(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Computes the similarity between two vectors, as measured by their cosine distance
        :param vec1:
        :param vec2:
        :return:
        """
        return cosine_similarity(vec1, vec2)
