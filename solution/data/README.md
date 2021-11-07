## Data

데이터 전처리, 후처리 등 데이터가 모델에 들어가기 전과 후의 처리에 대한 역할을 함

- [Processors Module](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors)
    - [core.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/core.py) : 데이터를 불러오는 역할을 하는 추상화 코드 구현
    - [corrupt.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/corrupt.py) : sentence permutation과 같이 데이터에 corruption을 주는 함수 구현
    - [mask.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/mask.py) : question 또는 context에 대해서 masking 해주는 함수 구현
    - [odqa.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/odqa.py) : ODQA Task에서 필요한 전처리 및 데이터 불러오는 함수 구현
    - [post.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/post.py) : 모델의 prediction에 대한 후처리 함수 구현
    - [prep.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/data/processors/prep.py) : extractive 등의 task에 따라 데이터셋을 features로 바꾸어 주는 함수 구현
