from datasets import Dataset
from elasticsearch import Elasticsearch, helpers

from solution.args import DataArguments
from solution.retrieval.elastic_engine.base import ElasticSearchBase


class ESRetrieval(ElasticSearchBase):
    
    def __init__(self, args: DataArguments):
        es = Elasticsearch(args.es_host_address)
        super().__init__(args, es)
        
    def retrieve(self, query_or_dataset, topk=1):
        doc_scores, doc_indices, doc_contexts = self.get_relevant_doc(query_or_dataset, topk)
        if isinstance(query_or_dataset, str):
            doc_scores = doc_scores[0]
            doc_indices = doc_indices[0]
            print("[Search query]\n", query_or_dataset, "\n")
            
            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.get(doc_indices[i]), end="\n\n")

            return (doc_scores, [self.get(doc_indices[i]) for i in range(topk)])
        
        elif isinstance(query_or_dataset, Dataset):
            cqas = self.get_dataframe_result(query_or_dataset,
                                             doc_scores,
                                             doc_indices,
                                             doc_contexts,
                                            )
            return cqas
        
        elif isinstance(query_or_dataset, list):
            return (doc_scores, doc_contexts)