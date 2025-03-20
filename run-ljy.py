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
# 打开并读取 JSON 文件
# with open('nq320k/dev.json', 'r', encoding='utf-8') as f1:
#     data_dict_1 = json.load(f1)

# # with open('nq320k/train.json.qg.json', 'r', encoding='utf-8') as f2:
# #     data_dict_2 = json.load(f2)

# # data_dict = data_dict_1 + data_dict_2
# # data_dict = data_dict
# df = pd.DataFrame(data_dict_1, columns=['Question', 'Id'])
# df["Id"] = df["Id"].astype(int)

# with open('nq320k/corpus_lite.json', 'r', encoding='utf-8') as c:
#     data_dict_c = json.load(c)

# df["Article"] = df["Id"].apply(lambda x: data_dict_c[x] if x < len(data_dict_c) else "No article available")
# # 保存为 Parquet
# df.to_parquet("nq320k/test.parquet", index=False)
# print(df)
train = pd.read_parquet("nq320k/train.parquet")
test = pd.read_parquet("nq320k/test.parquet")
print(train)
train_dataset = Dataset.from_dict(train)
test_dataset = Dataset.from_dict(test)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant rewrites relevant articles. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["Question"]},
        ],
    }
model_s = SentenceTransformer("all-MiniLM-L6-v2")

train_dataset = train_dataset.map(make_conversation)
print(train_dataset)
print(train_dataset[0]["prompt"])
test_dataset = test_dataset.map(make_conversation)
print(test_dataset)
train_dataset = train_dataset.remove_columns(["Id", "Question"])


model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

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





from sklearn.metrics.pairwise import cosine_similarity

def accuracy_reward(completions, **kwargs):
    """
    计算 completion 的 EOS 嵌入和目标 Article 文档的距离作为规则奖励
    """
    solutions = kwargs["Article"]
    # print(len(solutions))
    completion_contents = [completion[0]["content"] for completion in completions]
    # print(len(completion_contents))
    encoded_solutions = tokenizer(solutions, padding=True, truncation=True, return_tensors='pt').to("cuda")
    encoded_completion = tokenizer(completion_contents, padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        solutions_embeddings = model.get_input_embeddings()(encoded_solutions["input_ids"])#token embeddings
        completion_embeddings = model.get_input_embeddings()(encoded_completion["input_ids"])
        solutions_embeddings = mean_pooling(solutions_embeddings, encoded_solutions['attention_mask'])
        completion_embeddings = mean_pooling(completion_embeddings, encoded_completion['attention_mask'])
        solutions_embeddings = F.normalize(solutions_embeddings, p=2, dim=1)
        completion_embeddings = F.normalize(completion_embeddings, p=2, dim=1)
    similarity_score = cosine_similarity(solutions_embeddings, completion_embeddings, dim=1)
    rewards = similarity_score
    print(similarity_score)
    return rewards



# def accuracy_reward(completions, **kwargs):
#     """
#     计算 completion 的 EOS 嵌入和目标 Article 文档的距离作为规则奖励
#     """
#     solutions = kwargs["Article"]
#     # print(len(solutions))
#     completion_contents = [completion[0]["content"] for completion in completions]
#     # print(len(completion_contents))
    
#     solutions_embeddings = model_s.encode(solutions)
#     completion_embeddings = model_s.encode(completion_contents)
#     similarities = model_s.similarity(solutions_embeddings, completion_embeddings)
#     similarities = torch.diag(similarities).tolist()

#     rewards = similarities
    
#     return rewards

# def accuracy_reward(completions, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     solutions = kwargs["Article"]
#     completion_contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, solution in zip(completion_contents, solutions):
#         gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#         answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
#         if len(gold_parsed) != 0:
#             try:
#                 rewards.append(float(verify(answer_parsed, gold_parsed)))
#             except Exception:
#                 rewards.append(0.0)
#         else:
#             rewards.append(1.0)
#     return rewards

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model, reward_funcs=[format_reward, accuracy_reward], args=training_args, train_dataset=train_dataset)

trainer.train()
trainer.save_model(training_args.output_dir)