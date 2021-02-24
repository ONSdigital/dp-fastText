from sanic.blueprints import Blueprint

from dp4py_sanic.api.response import json

from dp_fasttext.api.request.fasttext_request import FasttextRequest


healthcheck_blueprint = Blueprint('healthchech', url_prefix='/healthcheck')


@healthcheck_blueprint.route('/', methods=['GET'])
async def health_check(request: FasttextRequest):
    """
    Empty healthcheck (nothing to check)
    :param request:
    :return:
    """
    return json(request, {}, 200)
