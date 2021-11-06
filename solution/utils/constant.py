from datasets import Sequence, Value, Features
from datasets import Dataset, DatasetDict


QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
ANSWER_COLUMN_NAME = "answers"

Q_ENCODER_NAME = "q_encoder"
P_ENCODER_NAME = "p_encoder"
COLBERT_NAME = "colbert"

MRC_EVAL_FEATURES = Features(
    {
        "answers": Sequence(
            feature={
                "text": Value(dtype="string", id=None),
                "answer_start": Value(dtype="int32", id=None),
            },
            length=-1,
            id=None,
        ),
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
    }
)


MRC_PREDICT_FEATURES = Features(
    {
        "context": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
    }
)
