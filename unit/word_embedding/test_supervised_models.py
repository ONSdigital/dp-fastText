import unittest
import fastText
from dp_fasttext.math_utils import cosine_sim


class TestSupervisedModels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSupervisedModels, self).__init__(*args, **kwargs)

        self.model = fastText.load_model("supervised_models/ons_supervised.bin")

    def test_cosine_sim(self):
        """
        Test the cosine similarity between two vectors
        :return:
        """
        u = self.model.get_sentence_vector("homicide")
        v = self.model.get_sentence_vector("murder")

        sim = cosine_sim(u, v)
        self.assertGreater(sim, 0.8, "Similarity must be greater than 0.8")
