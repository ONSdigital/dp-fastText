"""
Routes for querying supervised fastText model
"""
from sanic.blueprints import Blueprint

from dp4py_sanic.api.response.json_response import json

from dp_fasttext.ml.supervised import SupervisedModel
from dp_fasttext.app.fasttext_server import FasttextServer
from dp_fasttext.api.request.fasttext_request import FasttextRequest


supervised_blueprint = Blueprint('supervised', url_prefix='/supervised')


@supervised_blueprint.route('/labels', methods=['POST'])
def get_labels(request: FasttextRequest):
    """
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
