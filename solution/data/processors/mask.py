import os
import torch
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
from functools import partial
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .prep import get_extractive_features
from datasets import load_from_disk, DatasetDict, Dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from konlpy.tag import Mecab
import random
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_pickle(file_name, data_set):
    """Save dataset as a pickle file."""
    file=open(file_name,"wb")
    pickle.dump(data_set,file)
    file.close()
    
def cosine_similarity(A, B):
    """Calculate cosine similarity between A and B"""
    return dot(A, B) / (norm(A) * norm(B))

def get_masked_dataset_with_ST(train_data_path):
    """Masking on the context by using sentence transformer."""
    tokenizer = AutoTokenizer.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True)
    
    raw_train_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['train']
    raw_val_dataset = load_from_disk(os.path.join(train_data_path, "train_dataset"))['validation']
    column_names=raw_train_dataset.column_names
    
    
    _ext_prepare_train_features = partial(get_extractive_features, tokenizer=tokenizer, mode="train")
    
    tokenized_train_dataset = raw_train_dataset.map(
        _ext_prepare_train_features,
        batched=True,
        #num_proc=4,
        remove_columns=column_names,
    )
    
    _ext_prepare_validation_features = partial(get_extractive_features, tokenizer=tokenizer, mode="eval")
    
    tokenized_valid_dataset = raw_val_dataset.map(
        _ext_prepare_validation_features,
        batched=True,
        #num_proc=4,
        remove_columns=column_names,
    )
    
    masked_dataset = mask_span_unit(tokenized_train_dataset, tokenizer)
    
    new_dataset = DatasetDict({
        'train': masked_dataset,
        'validation': tokenized_valid_dataset
    })
    
    save_pickle("./test.pkl", new_dataset)
    
    return new_dataset

def make_word_dict(tokens, tokenizer, answer):
    """Check the token to make a perfect word."""
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
    """Calculate cosine similarity between query and 
       all words by using sentence transformer, 
       and mask top N words with high similarity."""
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
    """mask proper nouns and common nouns randomly included in the question."""
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
    
    save_pickle('./test11.pkl', new_dataset)
    
    return new_dataset


def make_trunc_dataset(examples, tokenizer):
    """Create dataset for random masking."""
    new_tokenized_ids=[]
    new_att =[]
    new_token_type=[]
    new_answer = []
    
    texts = [text['text'][0] for text in examples['answers']]
    
    tokenized_q = tokenizer(examples['question'], return_tensors='pt', truncation=True, max_length=384, padding="max_length")
    tokenized_c = tokenizer(examples['context'], return_tensors='pt', truncation=True, max_length=384, stride=128,return_overflowing_tokens=True,return_offsets_mapping=True,padding="max_length")
    tokenized_a = tokenizer(texts, return_tensors='pt', max_length=100, padding="max_length")
    
    sample_mapping = tokenized_c.pop("overflow_to_sample_mapping")
    
    for i in tqdm(sample_mapping):
        new_tokenized_ids.append(tokenized_q['input_ids'][i].tolist())
        new_att.append(tokenized_q['attention_mask'][i].tolist())
        new_token_type.append(tokenized_q['token_type_ids'][i].tolist())
        new_answer.append(tokenized_a['input_ids'][i].tolist())
    
    return { 'ids':new_tokenized_ids, 
            'attention':new_att, 
            'token_type':new_token_type, 
            'answer':new_answer }


def make_emb_dataset(dataset_path, mode):
    """Masks or adds words that the model we are going to use is confusing."""
    
    tokenizer = AutoTokenizer.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True)
    
    raw_train_dataset = load_from_disk(os.path.join(dataset_path, "train_dataset"))['train']
    raw_val_dataset = load_from_disk(os.path.join(dataset_path, "train_dataset"))['validation']
    
    tokenized_c = tokenizer(raw_train_dataset['context'], return_tensors='pt', truncation=True, max_length=384, stride=128,return_overflowing_tokens=True,return_offsets_mapping=True,padding="max_length")
    
    offset_mapping = tokenized_c.pop("offset_mapping")
    sample_mapping = tokenized_c.pop("overflow_to_sample_mapping")
    
    _make_trunc_dataset = partial(make_trunc_dataset, tokenizer = tokenizer)
    new_q = raw_train_dataset.map(
        _make_trunc_dataset,
        batched=True,
        remove_columns=raw_train_dataset.column_names
    )
    
    Tensor_dataset = TensorDataset(
        tokenized_c['input_ids'], tokenized_c['attention_mask'], tokenized_c['token_type_ids'],
        torch.tensor(new_q['ids']), torch.tensor(new_q['attention']), torch.tensor(new_q['token_type']),
        torch.tensor(new_q['answer'])
    )

    train_dataloader = DataLoader(
        Tensor_dataset,
        batch_size=2
    )
    
    if mode == "mask":
        masked_dataset = mask_to_word(train_dataloader, tokenizer, offset_mapping, sample_mapping, raw_train_dataset)
    elif mode == "hard":
        masked_dataset = make_harder_word(train_dataloader, tokenizer, offset_mapping, sample_mapping, raw_train_dataset)
    
    new_dataset = DatasetDict({
        'train':masked_dataset,
        'validation':raw_val_dataset
    })
    
    save_pickle("./test123.pkl", new_dataset)
    
    return new_dataset

def find_hard_word(tokenizer, ids, answer, idx):
    """find confusing words for adding"""
    
    front_idx = int(idx)
    back_idx = int(idx)
    
    tokens = tokenizer.convert_ids_to_tokens(ids)
    
    while True:
        if tokens[front_idx][:2] == "##":
            front_idx -= 1
        elif (len(tokens[front_idx])<=2 or tokens[front_idx][:2]!="##"):
            break
        else: 
            front_idx -= 1

    while True:
        if (len(tokens[back_idx+1])<=2) or (tokens[back_idx+1][:2]!="##"):
            break
        else:
            back_idx+=1
          
    word = re.sub('##','',''.join(tokens[front_idx:back_idx+1]))
    
    if answer in word:
        word=None
    
    return word

def find_word(tokenizer, ids, answer, idx):
    """find confusing words for masking"""
    front_idx = int(idx)
    back_idx = int(idx)
    
    tokens = tokenizer.convert_ids_to_tokens(ids)
    
    while True:
        if tokens[front_idx][:2] == "##":
            front_idx -= 1
        elif (len(tokens[front_idx])<=2 or tokens[front_idx][:2]!="##"):
            break
        else: 
            front_idx -= 1

    while True:
        if (len(tokens[back_idx+1])<=2) or (tokens[back_idx+1][:2]!="##"):
            break
        else:
            back_idx+=1
    
    word =  re.sub('##','',''.join(tokens[front_idx:back_idx+1]))
    
    if answer not in word:
        for idx in range(front_idx,back_idx+1):
            tokens[idx] = tokenizer.mask_token
    
    result = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    return result

def mask_to_word(dataloader, tokenizer, offset_mapping, sample_mapping, train_dataset):
    """find words that the model is confusing by using dot product and mask top N words."""
    new_ids=[]
    mask_token = tokenizer.mask_token_id
    
    ignore_tokens = [tokenizer.pad_token_id, 
                     tokenizer.unk_token_id,
                     tokenizer.cls_token_id, 
                     tokenizer.sep_token_id]
    
    
    p_encoder = AutoModel.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True).to(device)
    q_encoder = AutoModel.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True).to(device)
    
    torch.cuda.empty_cache()

    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            labels = batch[0].clone()
            
            p_inputs={
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device)
            }
            
            q_inputs = {
                "input_ids": batch[3].to(device),
                "attention_mask": batch[4].to(device),
                "token_type_ids": batch[5].to(device)
            }
            
            answers = [tokenizer.decode(i, skip_special_tokens=True) for i in batch[6]]
            
            matrix = torch.full(labels.shape, True)
            
            for ignore_token in ignore_tokens:
                ignore_mask = labels.eq(ignore_token)
                matrix.masked_fill_(ignore_mask, value=False)
            
            # 0 -> token 단위
            # 1 -> pooling layer 전체에 대한 embedding
            p_outputs = p_encoder(**p_inputs)[0]
            q_outputs = q_encoder(**q_inputs)[1]
            
            batch_size = p_outputs.shape[0]
            
            q_outputs = q_outputs.view(batch_size,1,-1)
            p_outputs = torch.transpose(p_outputs.view(batch_size, 384, -1), 1, 2)
            
            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
            sim_scores = sim_scores.view(batch_size, -1)
            sim_scores[~matrix] = -100
            sim_scores = F.log_softmax(sim_scores, dim=1)
            
            for idx, score in enumerate(sim_scores):
                sorted_score, sorted_idx = torch.sort(score, descending=True)
                for i in range(2):
                    p_inputs['input_ids'][idx] = find_word(tokenizer, p_inputs['input_ids'][idx], answers[idx], sorted_idx[i])
                new_ids.append(p_inputs['input_ids'][idx].tolist())
    
    origin_answers = train_dataset['answers']
    origin_start = [data['answer_start'][0] for data in train_dataset['answers']]
    origin_text = train_dataset['context']
    
    for list_idx, offset_list in enumerate(offset_mapping):
        for idx, offsets in enumerate(offset_list):
            if offsets[0] == offsets[1]:
                continue
            else:
                if new_ids[list_idx][idx] == tokenizer.mask_token_id:
                    for i in range(offsets[0], offsets[1]):
                        origin_list_idx = sample_mapping[list_idx]
                        origin_text[origin_list_idx] = list(origin_text[origin_list_idx])
                        origin_text[origin_list_idx][i] = '∬'
                        origin_text[origin_list_idx] = "".join(origin_text[origin_list_idx])

    for i, text in enumerate(origin_text):
        origin_text[i] = text[:origin_start[i]] + '[ans]' + text[origin_start[i]:]
    
    for idx, origin in enumerate(origin_text):
        origin_text[idx] = re.sub('∬+',tokenizer.mask_token, origin)
        origin_answers[idx]['answer_start'] = [origin_text[idx].find('[ans]')]
        origin_text[idx] = re.sub('\[ans\]','', origin_text[idx])
        
    return Dataset.from_dict({
        'title': train_dataset['title'],
        'context': origin_text,
        'question': train_dataset['question'],
        'id': train_dataset['id'],
        'answers': origin_answers,
        'document_id': train_dataset['document_id'],
        '__index_level_0__' : train_dataset['__index_level_0__'],
    })

def make_harder_word(dataloader, tokenizer, offset_mapping, sample_mapping, train_dataset):
    """find words that the model is confusing by using dot product and add top N words."""
    mask_token = tokenizer.mask_token_id
    
    ignore_tokens = [tokenizer.pad_token_id, 
                     tokenizer.unk_token_id,
                     tokenizer.cls_token_id, 
                     tokenizer.sep_token_id]
    
    
    p_encoder = AutoModel.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True).to(device)
    q_encoder = AutoModel.from_pretrained('kiyoung2/roberta-large-qaconv-sds', use_auth_token=True).to(device)
    
    torch.cuda.empty_cache()

    origin_answers = train_dataset['answers']
    origin_start = [data['answer_start'][0] for data in train_dataset['answers']]
    origin_text = train_dataset['context']
            
    for i, text in enumerate(origin_text):
        origin_text[i] = text[:origin_start[i]] + '[ans]' + text[origin_start[i]:]
        
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, batch in enumerate(tepoch):
            connect_idx = i*2
        
            labels = batch[0].clone()
            
            p_inputs={
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device)
            }
            
            q_inputs = {
                "input_ids": batch[3].to(device),
                "attention_mask": batch[4].to(device),
                "token_type_ids": batch[5].to(device)
            }
            
            answers = [tokenizer.decode(i, skip_special_tokens=True) for i in batch[6]]
            
            matrix = torch.full(labels.shape, True)
            
            for ignore_token in ignore_tokens:
                ignore_mask = labels.eq(ignore_token)
                matrix.masked_fill_(ignore_mask, value=False)
            
            p_outputs = p_encoder(**p_inputs)[0]
            q_outputs = q_encoder(**q_inputs)[1]
            
            batch_size = p_outputs.shape[0]
            
            q_outputs = q_outputs.view(batch_size,1,-1)
            p_outputs = torch.transpose(p_outputs.view(batch_size, 384, -1), 1, 2)
            
            sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()
            sim_scores = sim_scores.view(batch_size, -1)
            sim_scores[~matrix] = -100
            sim_scores = F.log_softmax(sim_scores, dim=1)
            
            for idx, score in enumerate(sim_scores):
                sorted_score, sorted_idx = torch.sort(score, descending=True)
                sim_words=[]
                
                for i in range(6):
                    find_w = find_hard_word(tokenizer, p_inputs['input_ids'][idx], answers[idx], sorted_idx[i])
                    if find_w is not None:
                        sim_words.append(find_w)
                        sim_words = list(set(sim_words))
                
                for word in sim_words:
                    origin_idx = sample_mapping[connect_idx]
                    word_idx = origin_text[origin_idx].find(word)
                    origin_text[origin_idx] = re.sub(re.escape(word), word*2, origin_text[origin_idx])
                    origin_text[origin_idx] = re.sub(re.escape(word*4), word*2, origin_text[origin_idx])
                
    for idx, origin in enumerate(origin_text):
        origin_answers[idx]['answer_start'] = [origin_text[idx].find('[ans]')]
        origin_text[idx] = re.sub('\[ans\]','', origin_text[idx])
        
    return Dataset.from_dict({
        'title': train_dataset['title'],
        'context': origin_text,
        'question': train_dataset['question'],
        'id': train_dataset['id'],
        'answers': origin_answers,
        'document_id': train_dataset['document_id'],
        '__index_level_0__' : train_dataset['__index_level_0__'],
    })