"""
Routes for querying supervised fastText model
"""
from sanic.blueprints import Blueprint

from dp4py_logging.time import timeit
from dp4py_sanic.api.response.json_response import json

from dp_fasttext.ml.supervised import SupervisedModel
from dp_fasttext.app.fasttext_server import FasttextServer
from dp_fasttext.api.request.fasttext_request import FasttextRequest


supervised_blueprint = Blueprint('supervised', url_prefix='/supervised')


@supervised_blueprint.route('/sentence/vector', methods=['POST'])
@timeit
async def get_sentence_vector(request: FasttextRequest):
    """
    Returns the vector for the input sentence
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: SupervisedModel = app.get_supervised_model()

    query: str = request.get_query_string()

    vector = model.get_sentence_vector(query)

    response = {
        "query": query,
        "vector": vector.tolist()
    }

    return json(request, response, 200)


@supervised_blueprint.route('/predict', methods=['POST'])
@timeit
async def predict(request: FasttextRequest):
    """
    TODO - batch requests
    Queries the supervised fastText model for learned labels
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: SupervisedModel = app.get_supervised_model()

    query: str = request.get_query_string()
    num_labels: int = request.get_num_labels()
    threshold: float = request.get_threshold()

    labels, probabilities = model.predict(query, k=num_labels, threshold=threshold)
    response = {
        "labels": labels,
        "probabilities": probabilities
    }

    return json(request, response, 200)
