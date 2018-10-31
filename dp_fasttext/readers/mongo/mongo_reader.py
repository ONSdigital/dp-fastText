from typing import List
from pymongo import MongoClient
from dp_fasttext.models.page import Page
from dp_fasttext.readers.reader import DocumentReader


class MongoReader(DocumentReader):
    def __init__(self, mongo_url: str='localhost', port: int=27017):
        self.client = MongoClient(mongo_url, port)

    @property
    def collection(self):
        return self.client.local.pages

    def load_pages(self) -> List[Page]:
        query = {
            "sections": {
                "$exists": True
            }
        }
        cursor = self.collection.find(query)

        pages = []
        for doc in cursor:
            pages.append(Page(doc))

        return pages
