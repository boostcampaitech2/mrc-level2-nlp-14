## Retrieval

- core.py : 가장 기본적인 retrieval 추상화 구현 및 메인 메서드와 그 외 필수 메서드 명시
- mixin.py : faiss indexing 사용과 retrieval output 을 만드는 mixin 을 각각 구현
- dense : dense retrieval 구현
    - base.py : retrieval의 필수 메서드를 dense retrieval 에 맞게 구현
    - colbert.py : colbert encoder 와 colbert retrieval 구현
    - dpr.py : bert encoder 와 DPR 구현
- elastic_engine : elastic engine 구현
    - api.py : Elasticsearch search를 위한 관련 기능 구현
    - base.py : Elasticsearh build를 위한 메서드 및 템플릿 구현
- sparse : sparse retrieval 구현
    - base.py : retrieval의 필수 메서드를 sparse retrieval 에 맞게 구현
    - bm25.py : sklearn base의 BM25 함수 구현
    - tfidf.py : sklearn base의 TF-IDF 함수 구현

### UML Diagram of Reader & Trainer
![Untitled](https://user-images.githubusercontent.com/88299729/140636333-efaea0c2-2030-4701-b06f-94c0954a91fe.png)
