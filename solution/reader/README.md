## Reader

- core.py : 가장 기본적인 reader 추상화 구현 및 메인 메서드와 그 외 필수 메서드 명시
- readers.py : extractive, generative, ensemble reader 정의
- trainers : reader 가 read 할 때 사용할 trainer 정의
    - base.py : mixin을 상속받는 base trainer 및 base seq2seq trainer 정의
    - ensemble.qa.py : question answering seq2seq trainer를 상속받는 question answering ensemble trainer 정의
    - mixin.py : optimizer 및 scheduler 의 추가 기능을 위한 mixin 구현
    - qa.py : base trainer를 상속받는 question answering trainer 정의
    - seq2seq_qa.py : base seq2seq trainer를 상속받는 question answering seq2seq trainer 정의
- architectures
    - models : 사용하는 모델 architectures를 구현
    - modeling_heads.py : model의 output head를 conv layer로 바꿀 수 있도록 구현
    - modeling_outputs.py : model의 ouput 값들을 담을 수 있는 dataclass 구현
    - modeling_utils.py : model head를 구성하는데 사용되는 기본적인 layers

## UML Diagram of Reader & Trainer
![Retreival   Reader Diagram drawio](https://user-images.githubusercontent.com/41335296/140636920-099510a5-b8e0-4920-bd07-71630f9973c6.png)
