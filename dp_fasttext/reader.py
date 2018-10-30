import abc
from typing import List

from dp_fasttext.page import Page


class DocumentReader(abc.ABC):

    @abc.abstractmethod
    def load_pages(self) -> List[Page]:
        pass