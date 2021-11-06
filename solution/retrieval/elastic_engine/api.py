from typing import Any, Dict
from datasets import Dataset
from elasticsearch import Elasticsearch, helpers

from solution.args import DataArguments
from .base import ElasticSearchBase


class ESRetrieval(ElasticSearchBase):

    def __init__(self, args: DataArguments):
        es = Elasticsearch(args.es_host_address,
                           timeout=args.es_timeout,
                           max_retries=args.es_max_retries,
                           retry_on_timeout=args.es_retry_on_timeout)
        super().__init__(args, es)

    def retrieve(self, query_or_dataset, topk=1, eval_mode=True) -> Any:
        """ Retrieve top-k documents using elastic search given dataset """

        results = self.get_relevant_doc(query_or_dataset, topk)
        doc_scores, doc_indices, doc_contexts = results

        if isinstance(query_or_dataset, str):
            doc_scores = doc_scores[0]
            doc_indices = doc_indices[0]
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" %
                      (i + 1, doc_scores[i]))
                print(self.get(doc_indices[i]), end="\n\n")

            return (doc_scores, [self.get(doc_indices[i]) for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            cqas = self.get_dataframe_result(query_or_dataset,
                                             doc_scores,
                                             doc_indices,
                                             doc_contexts,)
            return self.dataframe_to_dataset(cqas, eval_mode)

        elif isinstance(query_or_dataset, list):
            return (doc_scores, doc_contexts)

    def get(self, id) -> str:
        """ Get documents using id """

        doc = self.engine.get(index=self.index_name, id=id)
        return doc["_source"]["document_text"]

    @property
    def count(self) -> int:
        """ Return number of documents """
        return self.engine.count(index=self.index_name)["count"]

    def analyze(self, query) -> Any:
        """ Analyze query text usign analyzer tokenizer """

        body = {"analyzer": "my_analyzer", "text": query}
        return self.engine.indices.analyze(index=self.index_name, body=body)

    def make_query(self, query, topk) -> Dict:
        """ Given query and top-k parameter, make query dictionary used for retrieval """
        return {"query": {"match": {"document_text": query}}, "size": topk}

    def get_relevant_doc(self, query_or_dataset, topk) -> Any:
        """ Get relevant document using elastic search api """

        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        elif isinstance(query_or_dataset, str):
            query = [query_or_dataset]
        elif isinstance(query_or_dataset, list):
            query = query_or_dataset
        else:
            raise NotImplementedError

        body = []
        for i in range(len(query)*2):
            if i % 2 == 0:
                body.append({"index": self.index_name})
            else:
                body.append(self.make_query(query[i//2], topk))

        response = self.engine.msearch(body=body)["responses"]

        doc_scores = [[hit["_score"] for hit in res["hits"]["hits"]] for res in response]
        doc_indices = [[hit["_id"] for hit in res["hits"]["hits"]] for res in response]
        doc_contexts = [[hit["_source"]["document_text"] for hit in res["hits"]["hits"]] for res in response]

        return doc_scores, doc_indices, doc_contexts
