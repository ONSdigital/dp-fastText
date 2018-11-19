"""
Routes for querying supervised fastText model
"""
from sanic.blueprints import Blueprint
from sanic.exceptions import InvalidUsage

from dp4py_logging.time import timeit
from dp4py_sanic.api.response.json_response import json

from dp_fasttext.log import logger
from dp_fasttext.ml.supervised import SupervisedModel
from dp_fasttext.app.fasttext_server import FasttextServer
from dp_fasttext.api.request.fasttext_request import FasttextRequest


supervised_blueprint = Blueprint('supervised', url_prefix='/supervised')


@supervised_blueprint.route('/', methods=['GET'])
@timeit
async def info(request: FasttextRequest):
    """
    Returns info about the supervised model
    TODO - Add authentication
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: SupervisedModel = app.get_supervised_model()

    model_info = {
        "dimensions": model.get_dimension(),
        "isQuantised": model.is_quantized()
    }

    return json(request, model_info, 200)


@supervised_blueprint.route('vector', methods=['POST'])
@timeit
async def get_sentence_vector(request: FasttextRequest):
    """
    Returns the vector for the input sentence
    TODO - Add authentication
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: SupervisedModel = app.get_supervised_model()

    try:
        query: str = request.get_query_string()
    except InvalidUsage as e:
        logger.error(request.request_id, "Invalid request made to /supervised/sentence/vector", exc_info=e)
        return json(request, "Invalid request", e.status_code)

    vector = model.get_sentence_vector(query)

    response = {
        "query": query,
        "vector": vector.tolist()
    }

    return json(request, response, 200)


@supervised_blueprint.route('/vector/batch', methods=['POST'])
@timeit
async def batch_get_sentence_vector(request: FasttextRequest):
    """
    Batch process sentence vector requests
    :param request:
    :return:
    """
    app: FasttextServer = request.app
    model: SupervisedModel = app.get_supervised_model()

    try:
        queries: dict = request.get_batch_query_strings()
        logger.debug(request.request_id, "Got batch sentence vector request", extra={
            "params": {
                "queries": queries
            }
        })
    except InvalidUsage as e:
        logger.error(request.request_id, "Invalid request made to /supervised/sentence/vector/batch", exc_info=e)
        return json(request, "Invalid request", e.status_code)

    # Process
    results: dict = model.batch_get_sentence_vector(queries)

    response = {
        "results": results
    }

    # Return
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

    try:
        query: str = request.get_query_string()
        num_labels: int = request.get_num_labels()
        threshold: float = request.get_threshold()
    except InvalidUsage as e:
        logger.error(request.request_id, "Invalid request made to /supervised/predict", exc_info=e)
        return json(request, "Invalid request", e.status_code)

    labels, probabilities = model.predict(query, k=num_labels, threshold=threshold)
    response = {
        "labels": labels,
        "probabilities": probabilities
    }

    return json(request, response, 200)
