from dp4py_sanic.app.server import Server
from dp4py_sanic.unit.test_app import TestApp

from dp_fasttext.app.app import create_app


class FastTextTestApp(TestApp):
    def get_app(self) -> Server:
        return create_app()
