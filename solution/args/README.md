## args

사용하는 argument들을 용도에 따라 정리해 놓은 module

- [argparse.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/argparse.py) : argument들을 관리하는 파일에 따라 json, yaml file을 argument로 parsing 작업
- [base.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/base.py) : 가장 기본적으로 사용되는 Dataset, Model, Project arguments들에 대해서 추상화 코드 구현
- [data_args.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/data_args.py) : 데이터 로드 및 처리를 위한 argument 정의
- [model_args.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/model_args.py) : 모델 fine-tuning을 위한 model/config/tokenizer argument 정의
- [project_args.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/project_args.py) : wandb와 checkpoint 등 project 관리를 위한 argument 정의
- [training_args.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/args/training_args.py) : 학습 관리와 trainer 안에 들어가는 argument 정의
