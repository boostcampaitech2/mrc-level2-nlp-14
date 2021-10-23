### >>> 진규님 라이브러리 디펜던시
import os
import torch
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
from functools import partial
from solution.utils import ext_prepare_train_features, ext_prepare_validation_features
from datasets import load_from_disk, DatasetDict, Dataset
from tqdm.auto import tqdm
#from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from konlpy.tag import Mecab
import random
import re
### <<<

### >>> 명훈님 라이브러리 디펜던시
import numpy as np
### <<<

### >>> 진규님 작업 범위

def save_pickle(file_name, data_set):
    file=open(file_name,"wb")
    pickle.dump(data_set,file)
    file.close()
    
def cosine_similarity(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def get_masked_dataset(train_data_path):

    tokenizer = AutoTokenizer.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True)
    
    raw_train_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['train']
    raw_val_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['validation']
    column_names=raw_train_dataset.column_names
    print("data loaded")
    
    _ext_prepare_train_features = partial(ext_prepare_train_features, tokenizer=tokenizer)
    
    tokenized_train_dataset = raw_train_dataset.map(
        _ext_prepare_train_features,
        batched=True,
        #num_proc=4,
        remove_columns=column_names,
    )
    
    _ext_prepare_validation_features = partial(ext_prepare_validation_features, tokenizer=tokenizer)
    
    tokenized_valid_dataset = raw_val_dataset.map(
        _ext_prepare_validation_features,
        batched=True,
        #num_proc=4,
        remove_columns=column_names,
    )
    
    print("tokenized")
    
    masked_dataset = mask_span_unit(tokenized_train_dataset, tokenizer)
    
    new_dataset = DatasetDict({
        'train': masked_dataset,
        'validation': tokenized_valid_dataset
    })
    
    save_pickle("./test.pkl", new_dataset)
    
    return new_dataset

def make_word_dict(tokens, tokenizer, answer):
    word_start = False
    second_sep = False
    word_index = {}
    word = ''
    index = []
    
    for i, t in enumerate(tokens):
        if t == tokenizer.cls_token:
            continue
        elif t == tokenizer.sep_token:
            second_sep=True
            continue
        elif t == tokenizer.sep_token and second_sep:
            break
        elif t == tokenizer.pad_token:
            break
        
        if not t.startswith('#') and not word_start:
            word_start = True
            word += t
            index.append(i)
            if not tokens[i+1].startswith('#'):
                word_start = False
                if word != answer:
                    word_index[word.replace('#', '')] = index
                word = ''
                index = []
        if t.startswith('#') and word_start:
            word += t
            index.append(i)
            if i < 383 and (not tokens[i+1].startswith('#') or tokens[i+1] == tokenizer.sep_token):
                word_start = False
                if  word != answer:
                    word_index[word.replace('#', '')] = index
                word = ''
                index = []

    return word_index

def mask_span_unit(train_dataset, tokenizer):
    pad_idx = 0
    top_k = 20
    new_ids=[]
    
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    mask_token = tokenizer.mask_token_id
    
    for i, input_id in tqdm(enumerate(train_dataset['input_ids'])):
        sep_idx = np.where(np.array(input_id) == tokenizer.sep_token_id)[0][0]
        
        if tokenizer.pad_token_id in np.array(input_id):
            pad_idx = np.where(np.array(input_id) == tokenizer.pad_token_id)[0][0]
            
        question = tokenizer.decode(input_id[1:sep_idx]) # sep_idx[0][0]: 첫 번째 sep 토큰 위치
        answer  = tokenizer.decode(input_id[train_dataset['start_positions'][i]:train_dataset['end_positions'][i]+1])
    
        q_emb = model.encode(question)
        tokens = tokenizer.convert_ids_to_tokens(input_id)
        
        word_dict = make_word_dict(tokens, tokenizer, answer)
        
        sim_dict = {}
         
        for word in word_dict.keys():
            sim = cosine_similarity(q_emb,model.encode(word))
            
            if sim > 0.35:
                sim_dict[sim] = word_dict[word]
                
        sorted_sim = sorted(sim_dict.items(), reverse=True)
        
        span_to_mask=[]
        
        if len(sorted_sim) < top_k:
            for val in sorted_sim:
                span_to_mask.extend(val[1])
        else:
            for val in sorted_sim[:top_k]:
                span_to_mask.extend(val[1])
        
        for span_idx in list(span_to_mask):
            input_id[span_idx] = mask_token
        
        new_ids.append(input_id) 
    
    return Dataset.from_dict({
        'input_ids': new_ids,
        'end_positions': train_dataset['end_positions'],
        'attention_mask': train_dataset['attention_mask'],
        'start_positions': train_dataset['start_positions'],
    })

def make_question_random_masking(train_data_path):
    context_list = []
    question_list = []
    id_list=[]
    answer_list=[]
    
    tokenizer = AutoTokenizer.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True)
    
    train_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['train']
    val_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['validation']
    
    mecab = Mecab()
    for i in tqdm(range(train_dataset.num_rows)):
        text = train_dataset["question"][i]
        
        # 단어 기준 Masking
        for word, pos in mecab.pos(text):
            # 하나의 단어만 30% 확률로 Masking
            if pos in {"NNG", "NNP"} and (random.random() > 0.7):
                context_list.append(train_dataset["context"][i])
                question_list.append(re.sub(word, tokenizer.mask_token, text)) # tokenizer.mask_token
                id_list.append(train_dataset[i]["id"])
                answer_list.append(train_dataset[i]["answers"])
    
    # list를 Dataset 형태로 변환
    new_set = Dataset.from_dict({"id" : id_list,
                                        "context": context_list, 
                                        "question": question_list,
                                        "answers": answer_list})
    
    new_dataset = DatasetDict({
        'train': new_set,
        'validation': val_dataset
    })
    
    save_pickle('./test2.pkl', new_dataset)
    
    return new_dataset
        
### <<< 

### >>> 명훈님 작업 범위

### <<<