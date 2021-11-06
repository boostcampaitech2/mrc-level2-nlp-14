import re
import json
from typing import Dict, List, Union

import torch


class KoBpeTokenizer:
    """
    Korean BPE Tokenizer for PORORO NER Module
    """

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.cls_token = "<s>"
        self.sep_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    @property
    def cls_token_id(self):
        return self.vocab[self.bos_token]

    @property
    def sep_token_id(self):
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def nspecial(self):
        return 4

    @classmethod
    def from_file(cls, filename):
        """ Read from files for initialize KoBpeTokenizer object """
        with open(filename, "r") as f:
            vocab = json.load(f)
        return KoBpeTokenizer(vocab)

    def tokenize(self, text: str, add_special_tokens: bool = False) -> str:
        """ Tokenize single sentence wich metaspace """
        x = text.strip()
        x = [c for c in re.sub("\s+", " ", x)]
        result = list()
        for i in range(len(x)):
            if x[i] == " ":
                x[i+1] = f"▁{x[i+1]}"
                continue
            else:
                result.append(x[i])
        result[0] = f"▁{result[0]}"
        tokenized = " ".join(result)
        if add_special_tokens:
            tokenized = f"{self.cls_token} {tokenized} {self.sep_token}"
        return tokenized

    def decode(self, x: str) -> str:
        """ Decode tokens to text with metaspace """
        return x.replace(" ", "").replace("▁", " ").strip()

    def encode(self, text: Union[str, List[str]], **kwargs):
        """ Encode tokens to ids, used for single or batched sentence."""
        if isinstance(text, list):
            return self.encode_batch(text, **kwargs)
        add_special_tokens = kwargs.pop("add_special_tokens", False)
        return_tokens = kwargs.pop("return_tokens", False)
        tokenized = self.tokenize(text, add_special_tokens).split(" ")
        if return_tokens:
            return tokenized
        return [self.vocab.get(token, self.unk_token_id) for token in tokenized]

    def encode_batch(
        self,
        texts: List[str],
        return_tensors: bool = True,
        return_tokens: bool = False,
        **kwargs,
    ):
        """ Encode tokens to ids, used for batched sentences. """
        if return_tokens:
            return [
                self.encode(text, return_tokens=return_tokens, **kwargs)
                for text in texts
            ]
        encoded = [self.encode(text, **kwargs) for text in texts]
        max_seq_len = max(list(map(len, encoded)))
        return self.pad(encoded, max_seq_len, return_tensors=return_tensors)

    def pad(self, sequences: List[List[str]], max_seq_len, return_tensors: bool = True):
        """ Pad batched sequences. if return_tensors, then return torch.LongTensor object. """
        if return_tensors:
            padded = torch.LongTensor(
                [st + [self.pad_token_id] * (max_seq_len - len(st))
                 for st in sequences]
            )
        else:
            padded = [st + [self.pad_token_id] * (max_seq_len - len(st))
                      for st in sequences]
        return padded

    def __call__(self, text: Union[str, List[str]], **kwargs):
        """ Run encode method """
        return self.encode(text, **kwargs)
