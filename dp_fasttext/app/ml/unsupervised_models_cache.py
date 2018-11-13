"""
Cache for unsupervised ML models
"""
import logging

from dp_fasttext.ml.unsupervised import UnsupervisedModel

_cache = {}


def get_unsupervised_model(fname):
    """
    Initialise the supervised fastText .vec model
    :return:
    """
    if fname not in _cache:
        logging.info("Initialising unsupervised fastText model", extra={
            "model": {
                "filename": fname
            }
        })

        _cache[fname] = UnsupervisedModel(fname)

        logging.info("Successfully initialised unsupervised fastText model", extra={
            "model": {
                "filename": fname
            }
        })
    return _cache[fname]
