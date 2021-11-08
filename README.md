# Open-Domain Question Answering Solution

## 1. Introduction

<p align="center">
    <img src='https://github.com/boostcampaitech2/image-classification-level1-08/raw/master/_img/AI_Tech_head.png' height=50% width=50%></img>
</p>

<img src='https://github.com/boostcampaitech2/image-classification-level1-08/blob/master/_img/value_boostcamp.png?raw=true'></src>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 이 곳에는 그 세번 째 대회인 `Open-Domain Question Answering` 과제에 대한 **Level2-nlp-14조** 의 문제 해결 방법을 기록합니다.

### Team KiYOUNG2

_"Korean is all YOU Need for dialoGuE"_

#### 🔅 Members  

김대웅|김채은|김태욱|유영재|이하람|진명훈|허진규|
:-:|:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/41335296?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/60843683?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/47404628?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/53523319?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/35680202?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/37775784?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/88299729?v=4' height=80 width=80px></img>
[Github](https://github.com/KimDaeUng)|[Github](https://github.com/Amber-Chaeeunk)|[Github](https://github.com/taeukkkim)|[Github](https://github.com/uyeongjae)|[Github](https://github.com/hrxorxm)|[Github](https://github.com/jinmang2)|[Github](https://github.com/JeangyuHeo)

#### 🔅 Contribution  

- [`진명훈`](https://github.com/jinmang2) &nbsp; Project Management • Baseline Refatoring • Elastic Search • Masking • QA Convolution layer • Bart Denoising objective • Query Ensemble • Code Abstraction
- [`김대웅`](https://github.com/KimDaeUng) &nbsp; Curriculum Learning • DPR • Question Embedding Vis • KoEDA • Context Summary • Post processing • Ensemble(hard voting)
- [`김태욱`](https://github.com/taeukkkim) &nbsp; Data versioning • Elastic search • Retrieval experiment • N-gram Convolution layer • Bart Denoising objective • Curriculum Learning • Post processing
- [`허진규`](https://github.com/JeangyuHeo) &nbsp; Data versioning • Curriculum Learning • AEDA • Masking • Reader • EDA • Human Labeling
- [`이하람`](https://github.com/hrxorxm) &nbsp; Generative model • Extractive & Generative Ensemble • DPR • K-fold • Context Summary
- [`김채은`](https://github.com/Amber-Chaeeunk) &nbsp; Generative model • Extractive & Generative Ensemble • Underline Embedding Layer • Punctuation • Pivot Tanslation • Code • Data versioning • Context Summary
- [`유영재`](https://github.com/uyeongjae) &nbsp; Data versioning • Elastic search • Retrieval experiment • Data Augmentation • Post processing • Ensemble(hard & soft voting)

## 2. Project Outline
![mrc_logo](https://user-images.githubusercontent.com/37775784/140635905-748921a4-6b20-4cca-b3e4-24d894acfd6c.PNG)

**"서울의 GDP는 세계 몇 위야?", "MRC가 뭐야?"**

우리는 궁금한 것들이 생겼을 때, 아주 당연하게 검색엔진을 활용하여 검색을 합니다. 이런 검색엔진은 최근 MRC (기계독해) 기술을 활용하며 매일 발전하고 있는데요. 본 대회에서는 우리가 당연하게 활용하던 검색엔진, 그것과 유사한 형태의 시스템을 만들어 볼 것입니다.

**Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능**을 만드는 연구 분야입니다.다양한 QA 시스템 중, **Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는** 과정이 추가되기 때문에 더 어려운 문제입니다.

![odqa](https://user-images.githubusercontent.com/37775784/140635909-5508e825-472e-42cc-8c1c-69e0b4815c30.PNG)

**본 ODQA 대회에서 우리가 만들 모델은 two-stage**로 구성되어 있습니다. **첫 단계는 질문에 관련된 문서를 찾아주는 "retriever"** 단계이고, **다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader"** 단계입니다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 여러분들 손으로 직접 만들어보게 됩니다.

따라서, 대회는 더 정확한 답변을 내주는 모델을 만드는 팀이 좋은 성적을 거두게 됩니다.

![mrc_fig](https://user-images.githubusercontent.com/37775784/140635959-cf5951f3-3cb1-4e4b-94ed-0f6e7bed1996.png)

### 🏆 Final Score

![lb](https://user-images.githubusercontent.com/37775784/140636123-c6c8779b-d5b3-4bb8-955b-7f9c3ef44a5a.PNG)

## 3. Solution

기계 학습은 인간의 학습 방식에서 아이디어를 얻었습니다. 때문에 저희도 이번 ODQA 문제를 푸는 방향을 **사람과 같이 학습하는 모델 구축** 으로 잡았습니다. 사람과 같이 학습한다는 것을 정의하기 위해 저희는 아래와 같은 방안을 제시했습니다.
- 우리는 중요할 것이라 생각되는 부분에 밑줄을 긋는다 (Underlining)
- 초-중-고의 순으로 국가에서 정한 커리큘럼을 이수한다 (Curriculum Learning)
- 사람마다 학습을 위해 참고하는 자료가 다르다 (Data Augmentation)

실제로 초기 예측 구조를 구축한 다음 검증 데이터 세트에서 틀린 예제들을 분석한 결과, 저희는 아래와 같은 견해를 얻었습니다.
- Reader 문제] 날짜 문제를 잘 못 풀더라! → PORORO 모델의 기학습 가중치 활용 (날짜를 상대적으로 잘 맞춤)
- Reader 문제] 뒤에 조사가 붙은 채로 나오는 결과가 많더라! → 형태소 분석기 앙상블 활용
- Reader 문제] 복잡한 의미 관계 추론을 힘들어 하더라! → 다양한 데이터로 다양한 모델에 태워서 앙상블
- Retrieval 문제] 이상한 문서를 가져오더라! → Query 앙상블 + Title을 Context로 붙이기

저희는 위에서 얻은 견해를 기반으로 저희만의 solution을 4주 동안 개발하였으며 상세한 내용을 아래 발표 자료에 정리하였습니다.

- [1등 솔루션 발표 pdf](./assets/kiyoung2_odqa.pdf)

다양한 데이터 세트와 모델을 활용하고 학습 방식에도 curriculum learning 등을 통해 학습시킨 후에 앙상블을 했을 때 성능이 많이 올랐습니다.


## 4. How to Use
```
.
├── assets/kiyoung2_odqa.pdf
├── configs/examples.yaml
├── solution
│   ├── args/base.py
│   ├── data
│   │     ├── metrics/__init__.py
│   │     └── processors
│   │           ├── /core.py
│   │           ├── /corrupt.py
│   │           ├── /mask.py
│   │           ├── /odqa.py
│   │           ├── /post.py
│   │           └── /prep.py
│   ├── ner/core.py
│   ├── reader
│   │     ├── architectures/__init__.py
│   │     │     └── models/__init__.py
│   │     ├── trainers/base.py
│   │     ├── /core.py
│   │     └── /readers.py
│   ├── retrieval
│   │     ├── dense/base.py
│   │     ├── elastic_engine
│   │     │     ├── /api.py
│   │     │     └── /base.py
│   │     ├── sparse/base.py
│   │     ├── /core.py
│   │     └── /mixin.py
│   └── utils
├── .gitignore
├── README.md
└── new_run.py
```

아래 명령어로 실행 가능합니다.

```console
python new_run.py configs/examples.yaml
```

아래와 같이 모듈을 호출하여 사용할 수도 있습니다.
```python
import os
from solution.args import HfArgumentParser
from solution.args import (
    MrcDataArguments,
    MrcModelArguments,
    MrcTrainingArguments,
    MrcProjectArguments,
)
from solution.retrieval import RETRIEVAL_HOST

parser = HfArgumentParser(
    [MrcDataArguments,
     MrcModelArguments,
     MrcTrainingArguments,
     MrcProjectArguments]
)
args = parser.parse_yaml_file(yaml_file="configs/example.yaml")
data_args, model_args, training_args, project_args = args

data_args.dataset_path = "Write YOUR dataset path"
data_args.context_path = "Write YOUR context file name"
data_args.rebuilt_index = True

retriever = RETRIEVAL_HOST["elastic_engine"]["elastic_search"](data_args)
retrieve.retrieve("윤락행위등방지법이 전문 개정되었던 해는?")
```


## 5. References

### Paper
- [Kim et al., Document-Grounded Goal-Oriented Dialogue Systems on Pre-Trained Language Model with Diverse Input Representation, DialDoc 2021](https://aclanthology.org/2021.dialdoc-1.12.pdf)
- [Xu et al., Curriculum Learning for Natural Language Understanding, ACL 2020](https://aclanthology.org/2020.acl-main.542.pdf)
- [Omar Khattab and Matei Zaharia, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT, SIGIR 2020](https://arxiv.org/abs/2004.12832)

### Software
#### Reader
- [deepset-ai/xlm-roberta-large-squad2](https://huggingface.co/deepset/xlm-roberta-large-squad2)
- [klue/roberta-large](https://huggingface.co/klue/roberta-large)
- [huggingface/datasets](https://github.com/huggingface/datasets)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [kakaobrain/pororo](https://github.com/kakaobrain/pororo)

#### Retrieval
- [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)
- [elastricsearch](https://github.com/elastic/elasticsearch-py)
- [faiss](https://github.com/facebookresearch/faiss)
- [koreyou/bm25](https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8)
- [stranford-futuredata/ColBERT]( https://github.com/stanford-futuredata/ColBERT/tree/master/colbert)

#### Pre & Post processing
- [hyunwoongko/kss](https://github.com/hyunwoongko/kss)
- [konlpy](https://github.com/konlpy/konlpy)
- [khaiii](https://github.com/kakao/khaiii)
- [nltk](https://github.com/nltk/nltk)
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- [KoEDA](https://github.com/toriving/KoEDA)

#### ETC
- [mkorpela/overrides](https://github.com/mkorpela/overrides)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [pytorch](https://github.com/pytorch/pytorch)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [scipy](https://github.com/scipy/scipy)
