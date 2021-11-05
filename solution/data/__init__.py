from .data_collator import (
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)


DATA_COLLATOR = {
    "extractive": DataCollatorWithPadding,
    "generative": DataCollatorForSeq2Seq,
    "ensemble": DataCollatorForSeq2Seq,
}