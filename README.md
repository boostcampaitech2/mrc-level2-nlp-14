# Open-Domain Question Answering Solution

## 1. Introduction

<p align="center">
    <img src='https://github.com/boostcampaitech2/image-classification-level1-08/raw/master/_img/AI_Tech_head.png' height=50% width=50%></img>
</p>

<img src='https://github.com/boostcampaitech2/image-classification-level1-08/blob/master/_img/value_boostcamp.png?raw=true'></src>

본 과정은 NAVER Connect 재단 주관으로 인공지능과 딥러닝 Production의 End-to-End를 명확히 학습하고 실무에서 구현할 수 있도록 훈련하는 약 5개월간의 교육과정입니다. 전체 과정은 이론과정(U-stage, 5주)와 실무기반 프로젝트(P-stage, 15주)로 구성되어 있으며, 이 곳에는 그 세번 째 대회인 `Open-Domain Question Answering` 과제에 대한 **Level2-nlp-14조** 의 문제 해결 방법을 기록합니다.

### Team KiYOUNG2

"Korean is all YOU Need for dialoGuE"

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
- [`김채은`](https://github.com/Amber-Chaeeunk) &nbsp; Generative model • Extractive & Generative Ensemble • Underline Embedding Layer • Pivot Tanslation • Code • Data versioning • Context Summary
- [`유영재`](https://github.com/uyeongjae) &nbsp; Data versioning • Elastic search • Retrieval experiment • Data Augmentation • Post processing • Ensemble(hard & soft voting)

## Project Outline


## Solution


## How to Use
```
.
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
│   ├── utils
│
├── .gitignore
├── README.md
├── new_run.py
```

## References
