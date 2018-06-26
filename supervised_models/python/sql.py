import sqlite3
import logging
from tqdm import tqdm

import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def from_mongo(save_path):
    """
    Reads pages from mongo and stores in sqllite
    :return:
    """
    from supervised_models.python.mongo.mongo_reader import MongoReader

    reader = MongoReader()
    pages = reader.load_pages()

    documents = []
    for p in pages:
        docs = p.to_docs()
        for doc in docs:
            if doc is not None:
                documents.append(doc)

    store_contents(documents, save_path)


def store_contents(documents, save_path):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        documents: documents to store
        save_path: path to save db
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")
    count = 0

    with tqdm(total=len(documents)) as pbar:
        for doc in documents:
            c.execute("INSERT INTO documents VALUES (?,?)", (doc['id'], doc['text']))
            pbar.update()
            count+=1

    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()
