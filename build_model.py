import sys
import logging.config

from dp_fasttext.train import train_model
from dp_fasttext.corpa import generate_labelled_corpus, write_corpus

# Import readerrs
from dp_fasttext.mongo.mongo_reader import MongoReader
from dp_fasttext.elasticsearch.elasticsearch_reader import ElasticsearchReader

# Configure logging
from dp4py_sanic.logging.log_config import log_config

logging.config.dictConfig(log_config)


def main(corpus_prefix: str, output_fname:str, ndim: int, reader: str= 'mongo'):
    """

    :param corpus_prefix:
    :param output_fname:
    :param ndim:
    :param reader:
    :return:
    """
    if reader is not None:
        if reader == 'mongo':
            reader = MongoReader()
        elif reader == 'elasticsearch':
            reader = ElasticsearchReader()
        else:
            message = "Unknown reader type: %s" % reader
            logging.error(message)
            raise RuntimeError(message)

        pages = reader.load_pages()
        logging.info("Loaded %d pages" % len(pages))

        corpus = generate_labelled_corpus(pages)
        logging.info("Corpus contains %d lines" % len(corpus))

        write_corpus(corpus_prefix, corpus, randomize=True)

    model = train_model(corpus_prefix, output_fname, label_prefix='__label__', dim=ndim, minCount=500, minCountLabel=500,
                        verbose=1)

    valid_fname = "%s.valid" % corpus_prefix
    for k in [1, 5]:
        logging.info("Validating model", extra={
            "params": {
                "validation_file": valid_fname,
                "k": k
            }
        })
        N, P, R = model.test(valid_fname, k)
        logging.info("Test complete", extra={
            "number_of_samples": N,
            "precision_at_k=%d" % k: P,
            "recall_at_k=%d" % k: R
        })


def print_usage(argv):
    print("Usage (existing corpus): python %s <corpus_fname_prefix> <model_out_fname> <ndim>" % argv[0])
    print("Usage (build corpus + model): python %s <corpus_fname_prefix> <model_out_fname> <ndim> <reader_type>" % argv[0])
    exit()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_usage(sys.argv)

    corpus_fname_prefix = sys.argv[1]
    model_out_fname = sys.argv[2]
    ndim = int(sys.argv[3])

    reader_type = None
    if len(sys.argv) == 5:
        reader_type = sys.argv[4]

    main(corpus_fname_prefix, model_out_fname, ndim, reader=reader_type)

