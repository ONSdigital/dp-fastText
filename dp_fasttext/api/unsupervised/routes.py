"""
Routes for querying supervised fastText model
"""
from sanic.blueprints import Blueprint

from numpy import ndarray

from dp4py_logging.time import timeit
from dp4py_sanic.api.response.json_response import json

from dp_fasttext.ml.unsupervised import UnsupervisedModel

from dp_fasttext.app.fasttext_server import FasttextServer
from dp_fasttext.api.request.fasttext_request import FasttextRequest


unsupervised_blueprint = Blueprint('unsupervised', url_prefix='/unsupervised')


@unsupervised_blueprint.route('/similar/vector', methods=['POST'])
@timeit
async def get_similar_words(request: FasttextRequest):
    """
    Returns labels similar to the input vector
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: UnsupervisedModel = app.get_unsupervised_model()

    vector: ndarray = request.get_query_vector()
    num_labels: int = request.get_num_labels()

    # Get similar labels to the input vector
    words = model.similar_by_vector(vector, top_n=num_labels, return_similarity=False)

    # Return
    body = {
        "words": words
    }
    return json(request, body, 200)
