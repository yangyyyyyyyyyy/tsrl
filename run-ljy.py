
import json
import pandas as pd
from datasets import Dataset
import re
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoModelForCausalLM
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


train = pd.read_parquet("nq320k/train.parquet")
test = pd.read_parquet("nq320k/test.parquet")
train_dataset = Dataset.from_dict(train)
test_dataset = Dataset.from_dict(test)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant provides a better question. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the a better question. The reasoning "
    "process and better question are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer>better question here</answer>"
)
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["Question"]},
        ],
    }


train_dataset = train_dataset.map(make_conversation)
print(train_dataset)
print(train_dataset[0]["prompt"])
test_dataset = test_dataset.map(make_conversation)
print(test_dataset)
# train_dataset = train_dataset.remove_columns(["Id", "Article"])
# test_dataset = test_dataset.remove_columns(["Id", "Article"])

model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
re_tokenizer = AutoTokenizer.from_pretrained('infly/inf-retriever-v1-1.5b', trust_remote_code=True)
re_model = AutoModel.from_pretrained('infly/inf-retriever-v1-1.5b', device_map="auto",trust_remote_code=True)
lora_config = LoraConfig(
    task_type="CAUSAL_LM", 
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]


def extract_answer(content):
    match = re.search(r"<answer>(.*?)</answer>", content)    if not match:
        match = re.search(r"<answer>(.*)", content, re.DOTALL)
    return match.group(1).strip() if match else ""

# from sklearn.metrics.pairwise import cosine_similarity
copurs_embedding = torch.load("avg_sentence_embeddings.pt").to("cuda")

#r1的召回率
def accuracy_reward_rank(completions, **kwargs):
    solutions = kwargs["Id"]
    completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [extract_answer(content) for content in completion_contents]

    max_length = 8192
    batch_dict = re_tokenizer(completion_contents, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to('cuda')
    batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = re_model(**batch_dict)
        re_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        re_embeddings = F.normalize(re_embeddings, p=2, dim=1)

    re_similarity_score  = ( re_embeddings @ copurs_embedding.T) * 100
    max_scores, max_indices = torch.max(re_similarity_score, dim=1)

    topk_scores_1, topk_1 = torch.topk(re_similarity_score, k=1, dim=1)    
    topk_scores_10, topk_10 = torch.topk(re_similarity_score, k=10, dim=1)  
    topk_scores_100, topk_100 = torch.topk(re_similarity_score, k=100, dim=1)  
    gold_embeddings = copurs_embedding[solutions] # [bs, d]

    top1_embeddings = copurs_embedding[topk_1]     # [bs, 1, d]
    top10_embeddings = copurs_embedding[topk_10]   # [bs, 10, d]
    top100_embeddings = copurs_embedding[topk_100] # [bs, 100, d]

    sim_top1 = torch.einsum('bd,bkd->bk', gold_embeddings, top1_embeddings).squeeze(1)  # [bs]
    sim_top10 = torch.einsum('bd,bkd->bk', gold_embeddings, top10_embeddings)  # [bs, 10]
    sim_top100 = torch.einsum('bd,bkd->bk', gold_embeddings, top100_embeddings)  # [bs, 100]

    # max_scores, max_indices = similarity_scores.max(dim=1)
    reward = torch.full((gold_embeddings.size(0),), -1.0, device=gold_embeddings.device)


    is_top1 = torch.isclose(sim_top1, torch.tensor(1.0, device=sim_top1.device), atol=1e-5)
    reward[is_top1] = 1.0

    # 条件2：否则如果在 top10 中有 similarity==1（不包括 top1 的位置），奖励 reward_10
    in_top10 = torch.any(torch.isclose(sim_top10, torch.tensor(1.0, device=sim_top10.device), atol=1e-5), dim=1)
    reward[(~is_top1) & in_top10] = 0.8

    # 条件3：否则如果在 top100 中有 similarity==1，奖励 reward_100
    in_top100 = torch.any(torch.isclose(sim_top100, torch.tensor(1.0, device=sim_top100.device), atol=1e-5), dim=1)
    reward[(~is_top1) & (~in_top10) & in_top100] = 0.5
    
    return reward

def accuracy_reward_10(completions, **kwargs):
    solutions = kwargs["Id"]
    completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [extract_answer(content) for content in completion_contents]

    max_length = 8192
    batch_dict = re_tokenizer(completion_contents, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to('cuda')
    batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = re_model(**batch_dict)
        re_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        re_embeddings = F.normalize(re_embeddings, p=2, dim=1)

    re_similarity_score  = ( re_embeddings @ copurs_embedding.T) * 100
    topk_scores, topk_indices = torch.topk(re_similarity_score, k=10, dim=1)   #topk_indices是bs*k
    
    gold_embeddings = copurs_embedding[solutions] # [bs, d]
    # article_embeddings = copurs_embedding[max_indices]
    topk_embeddings = copurs_embedding[topk_indices]  # shape: [bs, k,d]
    # print(gold_embeddings.shape)
    # print(article_embeddings.shape)
    # similarity_score  = ( gold_embeddings @ article_embeddings.T).diagonal()
    similarity_scores = torch.einsum('bd,bkd->bk', gold_embeddings, topk_embeddings)
    
    max_scores, max_indices = similarity_scores.max(dim=1)

    reward = torch.isclose(max_scores, torch.tensor(1.0, device=max_scores.device), atol=1e-5).float()
    # print(similarity_score)
    return reward
    

def accuracy_reward_100(completions, **kwargs):
    solutions = kwargs["Id"]
    completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [extract_answer(content) for content in completion_contents]

    max_length = 8192
    batch_dict = re_tokenizer(completion_contents, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to('cuda')
    batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = re_model(**batch_dict)
        re_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        re_embeddings = F.normalize(re_embeddings, p=2, dim=1)

    re_similarity_score  = ( re_embeddings @ copurs_embedding.T) * 100
    topk_scores, topk_indices = torch.topk(re_similarity_score, k=100, dim=1)   #topk_indices是bs*k
    
    gold_embeddings = copurs_embedding[solutions] # [bs, d]
    # article_embeddings = copurs_embedding[max_indices]
    topk_embeddings = copurs_embedding[topk_indices]  # shape: [bs, k,d]
    # print(gold_embeddings.shape)
    # print(article_embeddings.shape)
    # similarity_score  = ( gold_embeddings @ article_embeddings.T).diagonal()
    similarity_scores = torch.einsum('bd,bkd->bk', gold_embeddings, topk_embeddings)
    max_scores, max_indices = similarity_scores.max(dim=1)
    reward = torch.isclose(max_scores, torch.tensor(1.0, device=max_scores.device), atol=1e-5).float()
    
    # reward = (max_scores ==1.0).float()
    # print(similarity_score)
    return reward


training_args = GRPOConfig(
    output_dir="Qwen2-7B-GRPO-test-reward_1_10_100_rank",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=128,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    per_device_train_batch_size=32,
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, accuracy_reward_rank], args=training_args, train_dataset=train_dataset)

trainer.train()
trainer.save_model(training_args.output_dir)
