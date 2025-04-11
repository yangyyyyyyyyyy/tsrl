import torch
import torch.nn.functional as F
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # 用于显示进度条
import pandas as pd
from datasets import Dataset

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 加载语料库（list of strings）
with open('nq320k/corpus_lite.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
model = AutoModel.from_pretrained('infly/inf-retriever-v1-1.5b', device_map="auto",trust_remote_code=True)

# 参数
max_length = 8192

def base_it(predict, label, at, score_func):
    assert len(predict) == len(label)
    scores = []
    for pred, lbs in zip(predict, label):
        pred = pred.tolist() if not isinstance(pred, list) else pred
        best_score = 0.
        if not isinstance(lbs, list):
            lbs = [lbs]
        for lb in lbs:
            if isinstance(lb, list):
                lb = lb[0]
            rank = pred[:at].index(lb) + 1 if lb in pred[:at] else 0
            cur_score = score_func(rank)
            best_score = max(best_score, cur_score)
        scores.append(best_score)
    return scores


def eval_recall(predict, label, at=10):
    scores = base_it(predict, label, at, lambda rank: int(rank != 0))
    return {f'R@{at}': sum(scores) / len(scores)}

copurs_embedding = torch.load("avg_sentence_embeddings.pt").to("cuda")

test = pd.read_parquet("nq320k/test_with_expanded_reward_1_10_100-90.parquet")
test_dataset = Dataset.from_dict(test)
queries = test_dataset["ExpandedQuestion"]

batch_size = 32
all_maxscores_index = []
for i in tqdm(range(0, len(queries), batch_size)):
    batch_texts = queries[i:i+batch_size]
    batch_dict = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to('cuda')
    batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings @ copurs_embedding.T) * 100
        topk = 100  # 或者你想评估的 R@K 值
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        all_maxscores_index.extend(topk_indices.tolist())
        # max_scores, max_indices = torch.max(scores, dim=1)
        # all_maxscores_index.extend(max_indices.tolist())
    
# all_maxscores_index 筛选出来的文章编号
print(len(all_maxscores_index))
labels = test_dataset["Id"]
result_1 = eval_recall(all_maxscores_index, labels, at=1)
result_10 = eval_recall(all_maxscores_index, labels, at=10)
result_100 = eval_recall(all_maxscores_index, labels, at=100)
print(result_1)  # 输出: {'R@3': 2 / 3 = 0.666...}
print(result_10)
print(result_100)
seen_split = json.load(open('nq320k/dev.json.seen'))
  
unseen_split = json.load(open('nq320k/dev.json.unseen'))
# print(eval_recall([all_maxscores_index[j] for j in seen_split], [labels[j] for j in seen_split],at=1))
# print(eval_recall([all_maxscores_index[j] for j in unseen_split], [labels[j] for j in unseen_split],at=1))





