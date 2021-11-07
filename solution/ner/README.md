## PORORO Named Entity Recognition

뽀로로 NER 모듈을 Batch 단위로 활용하기 위해 huggingface로 포팅한 모듈입니다.

- [bpetokenizer.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/ner/bpetokenizer.py): NER 모듈에서 사용하는 BPE Tokenizer 클래스를 담은 파일입니다.
- [core.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/ner/core.py): NER Module Interface 클래스를 담은 파일입니다. `from_pretrained` 메서드로 호출 가능합니다. 현재는 한국어만 지원합니다.
- [modeling_roberta.py](https://github.com/boostcampaitech2/mrc-level2-nlp-14/blob/main/solution/ner/modeling_roberta.py): NER 모듈에서 사용하는 Roberta Token Classifier 모델 클래스를 담은 파일입니다.
