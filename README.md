dp-fasttext
==================

# Configuration

### Environment variables

| Environment variable         | Default                                    | Description
| ---------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------
| TESTING                      | false                                      | Configures the app for unit testing.
| SUPERVISED_MODEL_FILENAME    | ./supervised_models/ons_supervised.bin     | Filename of supervised fastText model.
| UNSUPERVISED_MODEL_FILENAME  | ./supervised_models/ons_supervised.vec     | Filename of unsupervised fastText model.
| BIND_HOST                    | 0.0.0.0                                    | The host to bind to.
| BIND_PORT                    | 5100                                       | The port to bind to.
| SANIC_WORKERS                | 1                                          | Number of Sanic worker threads.
| ENABLE_PROMETHEUS_METRICS    | false                                      | Enable/disable the /metircs endpoint for prometheus.
| COLOURED_LOGGING_ENABLED     | false                                      | Enable/disable coloured logging.
| PRETTY_LOGGING               | false                                      | Enable/disable JSON formatting for logging.
| LOG_LEVEL                    | INFO                                       | Log level (INFO, DEBUG, or ERROR)

# Install

To install locally (not recommended), run ```make```. The code requires python3.6, and it is recommended that you setup 
a [virtual environment](https://docs.python.org/3/library/venv.html).
Alternatively (preferred approach), you can use the supplied Dockerfile to run in a container. When running with 
conceptual search and user recommendation enabled, the simplest approach is to use ```docker-compose``` with the
```docker-compose.yml``` provided to bring up dedicated instances of mongoDB and Elasticsearch. Note that for conceptual
search, the latter requires a [plugin for vector scoring](https://github.com/sully90/fast-elasticsearch-vector-scoring).  

# Running

There are two options for running the server:
Use ```python manager.py``` to use the internal Sanic server, or  ```./run_gunicorn.sh``` to initialise as a 
gunicorn server (supports multi-processing for multiple workers and threads per worker). This repository comes with a *test* [word2vec embeddings model](ml/data/word2vec/ons_supervised.vec) and [supervised model](unit/ml/test_data/supervised_models/ons_supervised.bin).

# Swagger

The swagger spec can be found in ```swagger.yaml```

# Testing

To run the unit tests, use: ```make test```.

### Licence

Copyright ©‎ 2016, Office for National Statistics (https://www.ons.gov.uk)

Released under MIT license, see [LICENSE](LICENSE.md) for details.

This software uses the fastText library, see [LICENSE](fasttext/LICENSE.md) for details.
