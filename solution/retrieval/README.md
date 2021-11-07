## Retrieval

- [core.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/core.py) : 가장 기본적인 retrieval 추상화 구현 및 메인 메서드와 그 외 필수 메서드 명시
- [mixin.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/mixin.py) : faiss indexing 사용과 retrieval output 을 만드는 mixin 을 각각 구현
- [dense](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/dense) : dense retrieval 구현
    - [base.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/dense/base.py) : retrieval의 필수 메서드를 dense retrieval 에 맞게 구현
    - [colbert.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/dense/colbert.py) : colbert encoder 와 colbert retrieval 구현
    - [dpr.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/dense/dpr.py) : bert encoder 와 DPR 구현
- [elastic_engine](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/elastic_engine) : elastic engine 구현
    - [api.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/elastic_engine/api.py) : Elasticsearch search를 위한 관련 기능 구현
    - [base.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/elastic_engine/base.py) : Elasticsearh build를 위한 메서드 및 템플릿 구현
- [sparse](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/sparse) : sparse retrieval 구현
    - [base.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/sparse/base.py) : retrieval의 필수 메서드를 sparse retrieval 에 맞게 구현
    - [bm25.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/sparse/bm25.py) : sklearn base의 BM25 함수 구현
    - [tfidf.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval/sparse/tfidf.py) : sklearn base의 TF-IDF 함수 구현

### UML Diagram of Retrieval
![Untitled](https://user-images.githubusercontent.com/88299729/140636333-efaea0c2-2030-4701-b06f-94c0954a91fe.png)
