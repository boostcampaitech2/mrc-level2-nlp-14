from abc import abstractmethod
from .retrieve_mixin import FaissMixin, PandasMixin


class RetrievalBase(FaissMixin, PandasMixin):
    pass


class SparseRetrieval(RetrievalBase):
    pass


class DenseRetrieval(RetrievalBase):
    pass