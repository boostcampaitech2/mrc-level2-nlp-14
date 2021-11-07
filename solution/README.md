## Solution

저희의 솔루션 모듈을 공개합니다! 아래의 폴더로 구성되어 있으며 폴더 내부에 있는 README 파일을 읽으시면 더욱 자세한 설명과 클래스 다이어그램을 확인하실 수 있습니다.

- [args](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/args): 각 모듈에서 사용될 인자들을 추상화된 클래스로 관리합니다. `./configs/examples.yaml` 파일로 어떤 인자를 사용할 지 컨트롤할 수 있습니다. 자세한 설명은 파일 내부의 metadata의 help text를 확인해주세요.
- [data](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/data): 각 모듈에 공급될 데이터 셋에 적용되는 모듈을 모아뒀습니다. 아래의 기능들을 수행합니다.
    1. 데이터셋 호출
    2. 전처리 적용
    3. Trainer에 전달할 후처리 함수 상세
    4. Corrupt functions
    5. 결과를 평가할 metric fn
    6. data collator
- [ner](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/ner): 데이터 처리에 사용되는 Named Entity Recognition 모듈입니다.
- [reader](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/reader): 2-stage 모델 중 MRC에 해당하는 `Reader` 클래스를 담은 모듈입니다. `read`라는 main method를 가지며 🤗 transformers의 Trainer 객체의 train, evaluate, predict 기능을 수행합니다. Trainer 객체에서 사용될 아키텍쳐에 대한 소스 코드 또한 담고 있습니다.
- [retrieval](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/retrieval): 2-stage 모델 중 검색에 해당하는 `Retrieval` 클래스를 담은 모듈입니다. `retrieve`라는 main method를 가지며 ElasticSearch, Dense, Sparse 엔진 3가지를 활용하실 수 있습니다. 특히, 저희의 아이디어 중 `Underlining`이 구현된 모듈입니다.
- [utils](https://github.com/boostcampaitech2/mrc-level2-nlp-14/tree/main/solution/utils): 각 모듈에서 활용될 기타 기능들을 모아둔 모듈입니다.
