from elasticsearch import Elasticsearch, helpers

from solution.args import DataArguments
from solution.retrieval.core import SearchBase


class ElasticSearchBase(SearchBase):
    
    def __init__(self, args: DataArguments, es: Elasticsearch):
        super().__init__(args)
        self.es = es
        if self.es.indices.exists(index=self.index_name):
            self.delete(self.index_name)
        self.build_index()
        assert self.es.ping(), "Fail ping."
        
    def build_index(self, index_name: str):
        print(f"Create elasticsearch index: {index_name}")
        self.es.indices.create(index=index_name, body=self.index_config, ignore=400)
        document_texts = [
            {"_index": self.index_name, 
             "_source": {"document_text" : doc}}
            for doc in self.contexts
        ]
        helpers.bulk(es, docs)
    
    def delete(self, index_name: str):
        print(f"Delete elasticsearch index: {index_name}")
        self.es.indices.delete(index=index_name, ignore=["400", "404"])
    
    @property
    def index_name(self):
        return self.args.index_name
    
    @index_name.setter
    def index_name(self, index_name: str):
        self.args.index_name = index_name
        
    @property
    def index_config(self):
        index_config = {"settings": {}, "mappings": {}}
        index_config = self.update_settings(index_config)
        index_config = self.update_mappings(index_config)
        return index_config
    
    def update_settings(self, index_config):
        _analyzer = {
            "my_analyzer": {
                "type": "custom", 
                "tokenizer": "nori_tokenizer", 
                "decompound_mode": self.args.decompound_mode,
            }
        }
        if self.args.use_korean_stopwords:
            _analyzer["my_analyzer"].update({"stopwords": "_korean_"})
        if self.args.use_korean_synonyms:
            _analyzer["my_analyzer"].update({"synonyms": "_korean_"})
        
        _filter = {}
        es_filter = []
        # 영문 소문자 변환
        if self.args.lowercase:
            es_filter += ["lowercase"]
        # 사용자 설정 stopword
        if self.args.stopword_path:
            stop_filter = {
                "type": "stop", 
                "stop_words_path": self.args.stopword_path
            }
            _filter.update({"stop_filter": stop_filter})
            es_filter += ["stop_filter"]
        # 한자 음독 변환
        if self.args.nori_readingform:
            es_filter += ["nori_readingform"]
        # chinese-japanese-korean bigram
        if self.args.cjk_bigram:
            es_filter += ["cjk_bigram"]
        # 아라비아 숫자 외에 문자를 전부 아라비아 숫자로 변경
        if self.args.decimal_digit:
            es_filter += ["decimal_digit"]
        _analyzer["my_analyzer"].update({"filter": es_filter})
        
        all_sim_kwargs = {
            "bm25_similarity": {
                'type':'BM25',
                'b': self.args.b,
                'k1': self.args.k1,
            },
            "dfr_similarity": {
                'type':'DFR',
                "basic_model": self.args.dfr_basic_model,
                "after_effect": self.args.dfr_after_effect,
                "normalization": self.args.es_normalization,
            },
            "dfi_similarity": {
                'type':'DFI',
                "independence_measure": self.args.dfi_measure,
            },
            "ib_similarity": {
                'type':'IB',
                "distribution": self.args.ib_distribution,
                "lambda": self.args.ib_lambda,
                "normalization": self.args.es_normalization,
            },
            # LM Dirichlet
            "lmd_similarity": {
                'type':'LMDirichlet',
                "mu": self.args.lmd_mu,
            },
            # LM Jelinek Mercer
            "lmjm_similarity": {
                'type':'LMJelinekMercer',
                "lambda": self.args.lmjm_lambda,
            },
        }
        sim_kwargs = all_sim_kwargs[self.args.es_similarity]
        
        assert self.args.es_similarity in sim_kwargs.keys()
            
        index_config["settings"] = {
            "analysis": {"filter": _filter, "analyzer": _analyzer},
            "similarity": {"type": self.args.es_similarity, **sim_kwargs},
        }
        return index_config
    
    def update_mappings(self, index_config):
        index_config["mappings"] = {
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic.html
            "dynamic": "strict", # 새 필드가 감지되면 예외가 발생하고 문서가 거부됨.
            "properties": {
                "document_text": {"type": "text", "analyzer": "my_analyzer",
                                  "similarity": self.args.es_similarity}
            }
        }
        return index_config
    
    def get(self, id):
        doc = self.es.get(index=self.index_name, id=id)
        return doc["_source"]["document_text"]
    
    @property
    def count(self):
        return es.count(index=self.index_name)["count"]
    
    def analyze(self, query):
        body = {"analyzer": "my_analyzer", "text": query}
        return self.es.indices.analyze(index=self.index_name, body=body)
    
    def make_query(self, query, topk):
        return {"query": {"match": {"document_text": query}}, "size": topk}
    
    def get_relevant_doc(self, query, topk):
        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        else:
            query = [query]
                    
        body = [
            make_query(query[i//2]) if i % 2 else {"index": self.index_name}
            for i in range(len(query)*2)
        ]
        response = es.msearch(body=body)["response"]
        
        doc_scores = [[hit["_score"] for hit in res["hits"]["hits"]] for res in response]
        doc_indices = [[hit["_id"] for hit in res["hits"]["hits"]] for res in response]
        
        return doc_scores, doc_indices