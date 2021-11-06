import time
import json
from typing import Any
from datasets import Dataset
from elasticsearch import Elasticsearch, helpers

from solution.args import MrcDataArguments
from ..core import SearchBase


class ElasticSearchBase(SearchBase):

    def __init__(self, args: MrcDataArguments, es: Elasticsearch):
        super().__init__(args)
        self.engine = es
        self._index_config = None
        if args.rebuilt_index and self.is_exists_index():
            print("Rebuild index...")
            self.delete(self.index_name)
            self.args.rebuilt_index = False
        if not self.is_exists_index():
            self.build_index(self.index_name)
        if self.index_config is None:
            with open("configs/es_index_config.json", "r", encoding="utf-8") as f:
                self.index_config = json.load(f)
        assert self.engine.ping(), "Fail ping."

    def build_index(self, index_name: str):
        """ Build elastic search index using bulk api """

        assert not self.is_exists_index()
        t0 = time.time()
        print(f"Create elasticsearch index: {index_name}")
        index_config = self.build_index_config()
        self.engine.indices.create(
            index=index_name, body=index_config, ignore=400)
        document_texts = [
            {"_id": i,
             "_index": self.index_name,
             "_source": {"document_text": doc}}
            for i, doc in enumerate(self.contexts)
        ]
        helpers.bulk(self.engine, document_texts)
        with open("configs/es_index_config.json", "w", encoding="utf-8") as f:
            json.dump(index_config, f)
        print(f"Done {time.time() - t0:.3}")

    def delete(self, index_name: str):
        """ Delete elastic search index """

        print(f"Delete elasticsearch index: {index_name}")
        self.engine.indices.delete(index=index_name, ignore=["400", "404"])

    @property
    def index_name(self) -> str:
        """ Get index name from arguments"""
        return self.args.index_name

    @index_name.setter
    def index_name(self, index_name: str):
        """ Set index name to arguments"""
        self.args.index_name = index_name

    @property
    def indices(self) -> Any:
        """ Get current index from elastic search engine """
        return self.engine.indices

    def is_exists_index(self) -> bool:
        """ Returns whether the index already exists """
        return self.engine.indices.exists(index=self.index_name)

    @property
    def index_config(self):
        """ Get index configuration """
        # 엘라스틱 서치에서 받아오는 메서드 존재함.
        return self._index_config

    @index_config.setter
    def index_config(self, val):
        """ Set index configuration """

        self._index_config = val

    def build_index_config(self):
        """ Build index configuration using data arguments """
        index_config = {"settings": {}, "mappings": {}}
        index_config = self.update_settings(index_config)
        index_config = self.update_mappings(index_config)
        return index_config

    def update_settings(self, index_config):
        """ Update index config's `settings` contents using DataArguments """

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
                'type': 'BM25',
                'b': self.args.b,
                'k1': self.args.k1,
            },
            "dfr_similarity": {
                'type': 'DFR',
                "basic_model": self.args.dfr_basic_model,
                "after_effect": self.args.dfr_after_effect,
                "normalization": self.args.es_normalization,
            },
            "dfi_similarity": {
                'type': 'DFI',
                "independence_measure": self.args.dfi_measure,
            },
            "ib_similarity": {
                'type': 'IB',
                "distribution": self.args.ib_distribution,
                "lambda": self.args.ib_lambda,
                "normalization": self.args.es_normalization,
            },
            # LM Dirichlet
            "lmd_similarity": {
                'type': 'LMDirichlet',
                "mu": self.args.lmd_mu,
            },
            # LM Jelinek Mercer
            "lmjm_similarity": {
                'type': 'LMJelinekMercer',
                "lambda": self.args.lmjm_lambda,
            },
        }
        sim_kwargs = all_sim_kwargs[self.args.es_similarity]

        assert self.args.es_similarity in all_sim_kwargs.keys()

        index_config["settings"] = {
            "analysis": {"filter": _filter, "analyzer": _analyzer},
            "similarity": {
                self.args.es_similarity: {
                    "type": self.args.es_similarity,
                    **sim_kwargs,
                },
            }
        }
        return index_config

    def update_mappings(self, index_config):
        """ Update index config's `mappings` contents using DataArguments """

        index_config["mappings"] = {
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic.html
            "dynamic": "strict",  # 새 필드가 감지되면 예외가 발생하고 문서가 거부됨.
            "properties": {
                "document_text": {"type": "text", "analyzer": "my_analyzer",
                                  "similarity": self.args.es_similarity}
            }
        }
        return index_config
