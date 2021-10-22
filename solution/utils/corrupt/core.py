### >>> 진규님 라이브러리 디펜던시
import os
import torch
import pickle
import numpy as np
from solution.utils import ext_prepare_train_features, ext_prepare_validation_features
from datasets import load_from_disk, DatasetDict
from tqdm.auto import tqdm
from torch.nn.functional import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
### <<<

### >>> 명훈님 라이브러리 디펜던시
import numpy as np
### <<<

### >>> 진규님 작업 범위

def save_pickle(data_set):
    file=open('./test.pkl',"wb")
    pickle.dump(data_set,file)
    file.close()
    
def get_masked_dataset(train_data_path):
    tokenizer = AutoTokenizer.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True)
    
    raw_train_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['train']
    raw_val_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['validation']
    
    print("data loaded")
    
    tokenized_train_dataset = ext_prepare_train_features(raw_train_dataset, tokenizer)
    tokenized_valid_dataset = ext_prepare_validation_features(raw_val_dataset, tokenizer)
    
    print("tokenized")
    
    masked_dataset = mask_span_unit(tokenized_train_dataset, tokenizer)
    
    new_dataset = DatasetDict({
        'train': masked_dataset,
        'validation': tokenized_valid_dataset
    })
    
    save_pickle(new_dataset)
    
    return new_dataset

def make_word_dict(tokens, tokenizer, answer):
    word_start = False
    word_index = {}
    word = ''
    index = []
    
    for i, t in enumerate(tokens):
        if t == tokenizer.cls_token:
            continue
        elif t == tokenizer.sep_token:
            break
        if t.startswith('▁') and not word_start:
            word_start = True
            word += t
            index.append(i)
            if tokens[i+1].startswith('▁'):
                word_start = False
                if word not in answer:
                    word_index[word.replace('▁', '')] = index
                word = ''
                index = []
        if not t.startswith('▁') and word_start:
            word += t
            index.append(i)
            if i < 383 and (tokens[i+1].startswith('▁') or tokens[i+1] == sep_token):
                word_start = False
                if word_ not in answer:
                    word_index[word.replace('▁', '')] = index
                word = ''
                index = []

    return word_index

def mask_span_unit(train_dataset, tokenizer):
    pad_idx = 0
    top_k = 20
    print("in")
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    mask_token = tokenizer.mask_token_id
    
    for i, input_id in tqdm(enumerate(train_dataset['input_ids'])):
        print('working')
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
    
        train_dataset['input_ids'][i] = input_id
    
    print(train_dataset['input_ids'][:5])
    
    return train_dataset
        
### <<< 

### >>> 명훈님 작업 범위

### <<<