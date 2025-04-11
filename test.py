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
from tqdm import tqdm  # 用于显示进度条



test = pd.read_parquet("nq320k/test.parquet")
test_dataset = Dataset.from_dict(test)
print(test_dataset)
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


test_dataset = test_dataset.map(make_conversation)

# test_dataset = test_dataset.remove_columns(["Id", "Article"])

model_id = "Qwen2-7B-GRPO-test-reward_1_10_100_rank/checkpoint-300"
trained_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
trained_tokenizer = AutoTokenizer.from_pretrained(model_id)


import time


# def generate_with_reasoning(prompt):
#     # Build the prompt from the dataset
#     prompt = " ".join(entry["content"] for entry in prompt)

#     # Tokenize and move to the same device as the model
#     inputs = trained_tokenizer(prompt, return_tensors="pt").to(trained_model.device)

#     # Generate text without gradients
#     start_time = time.time()
#     with torch.no_grad():
#         output_ids = trained_model.generate(**inputs, max_length=500)
#     end_time = time.time()

#     # Decode and extract model response
#     generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # Get inference time
#     inference_duration = end_time - start_time

#     # Get number of generated tokens
#     num_input_tokens = inputs["input_ids"].shape[1]
#     num_generated_tokens = output_ids.shape[1] - num_input_tokens

#     return generated_text, inference_duration, num_generated_tokens
def generate_batch_with_reasoning(batch_prompts):
    # 构造每条 prompt 的完整文本
    prompts_text = [" ".join(entry["content"] for entry in prompt) for prompt in batch_prompts]

    # 批量 tokenizer
    inputs = trained_tokenizer(prompts_text, return_tensors="pt", padding=True, truncation=True).to(trained_model.device)

    # 推理
    with torch.no_grad():
        output_ids = trained_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            do_sample=False,
            num_beams=1,
        )

    # 解码
    generated_texts = trained_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return generated_texts

batch_size= 32
all_prompts = test_dataset["prompt"]
expanded_questions = []
for i in tqdm(range(0, len(all_prompts), batch_size)):
    batch_prompts = all_prompts[i:i + batch_size]
    generated_texts = generate_batch_with_reasoning(batch_prompts)

    for text in generated_texts:
        answers = re.findall(r'<answer>(.*?)</answer>', text)
        last_answer = answers[-1] if answers else None
        # print(f"完整输出长度: {len(text)}")
        # print(f"匹配答案: {answers}")
        # print(f"最终答案: {last_answer}")
        expanded_questions.append(last_answer)

# assert len(test_dataset) == len(expanded_questions), "数量不匹配，请检查生成逻辑"        
test_dataset = test_dataset.add_column("ExpandedQuestion", expanded_questions)

test_dataset.to_parquet("nq320k/test_with_expanded_reward_1_10_100-300.parquet")




