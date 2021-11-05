import os
import sys
import faiss
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer
from solution.utils.constant import (
    MRC_EVAL_FEATURES,
    MRC_PREDICT_FEATURES
)
from tqdm import tqdm


class FaissMixin:
    
    def build_faiss(self, data_path: str, num_clusters: int = 64):
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.
        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """
        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")
        

class OutputMixin:
    
    def get_dataframe_result(
        self,
        query_or_dataset, 
        doc_scores,
        doc_indices,
        doc_contexts=None,
    ) -> pd.DataFrame:
        """
        Retrieval 결과를 DataFrame으로 정리하여 반환합니다.
        """
    
        #arguement수정 필요
        if self.args.do_punctuation == False:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.punct_model_name_or_path,
                use_auth_token=self.args.punct_use_auth_token, 
                revision=self.args.punct_revision
                )
        
            sentence_encoder = AutoModel.from_pretrained(
                self.args.punct_model_name_or_path, 
                use_auth_token=self.args.punct_use_auth_token, 
                revision=self.args.punct_revision
                ).to(device)
        
        total = []
        for idx, example in enumerate(tqdm(query_or_dataset)):
            if doc_contexts:
                contexts = doc_contexts[idx]
            else:
                contexts = [self.contexts[pid] for pid in doc_indices[idx]]
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, score, context를 반환합니다.
                "context_id": doc_indices[idx],
                "context_score": doc_scores[idx],
                "context": self.process_topk_context(contexts, example["question"], sentence_encoder, tokenizer, device)
            }
            # print(tmp['context'])
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)
     
        return pd.DataFrame(total)
    

    def dataframe_to_datasetdict(
        self,
        df: pd.DataFrame,
        eval_mode: bool = True,
    ) -> DatasetDict:
        features = MRC_EVAL_FEATURES if eval_mode else MRC_PREDICT_FEATURES
        datasets = DatasetDict(
            {"validation": Dataset.from_pandas(df, features=features)}
        )
        return datasets

    def dataframe_to_dataset(
        self,
        df: pd.DataFrame,
        eval_mode: bool = True,
    ) -> Dataset:
        features = MRC_EVAL_FEATURES if eval_mode else MRC_PREDICT_FEATURES
        datasets = Dataset.from_pandas(df, features=features)
        return datasets


    def process_topk_context(self, contexts, question, sentence_encoder, tokenizer, device):
        if self.args.do_punctuation == False:
            contexts = "#".join(contexts)
            contexts = contexts.split('#')
            contexts = [context.split('[TITLE]')[-1] if '[TITLE]' in context else context for context in contexts]
            return " ".join(contexts)
        else:
            q_seqs = tokenizer(
                question, 
                padding="max_length", 
                truncation=True, 
                max_seq_length=self.args.punct_max_seq_length, 
                return_tensors='pt'
                )
            p_seqs = tokenizer(
                contexts, 
                padding="max_length", 
                truncation=True, 
                max_seq_length=self.args.punct_max_seq_length, 
                return_tensors='pt'
                )

            torch.cuda.empty_cache()

            p_inputs = {'input_ids': p_seqs['input_ids'].to(device),
            'attention_mask': p_seqs['attention_mask'].to(device),
            'token_type_ids': p_seqs['token_type_ids'].to(device)
            }
        
            q_inputs = {'input_ids': q_seqs['input_ids'].to(device),    
            'attention_mask': q_seqs['attention_mask'].to(device),
            'token_type_ids': q_seqs['token_type_ids'].to(device)}
            
            sentence_encoder.eval()

            with torch.no_grad():
                p_outputs = sentence_encoder(**p_inputs)
                q_outputs = sentence_encoder(**q_inputs)

            dot_prod_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
            rank = torch.self.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
            topk_sentences = rank[:self.args.top_k_punctuation].tolist()

            new_contexts = []
            for i, sentence in enumerate(contexts): 
                if i in topk_sentences:
                    sentence = '^' + sentence + '※'
                    new_contexts.append(sentence)
                else:
                    new_contexts.append(sentence)

            return ' '.join(new_contexts)