
import copy
from abc import ABC
import gc
from transformers import T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
from torch.nn.utils.rnn import pad_sequence
from transformers import Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.generation import GenerationMixin
from torch import nn, Tensor
import torch.distributed as dist
from typing import Optional, Union, List, Dict, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F
from ios import read_pkl, write_pkl, read_file
from collections import defaultdict
from copy import deepcopy
import numpy as np
import json
import faiss
import torch
import os
import argparse
import time
from tqdm import tqdm
import torch



@dataclass
class QuantizeOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    all_dense_embed: Optional[torch.FloatTensor] = None
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None
    probability: Optional[torch.FloatTensor] = None
    code_logits: Optional[torch.FloatTensor] = None



@torch.no_grad()
def sinkhorn_raw(out: Tensor, epsilon: float,
                 sinkhorn_iterations: int,
                 use_distrib_train: bool):
    # out是经过归一化之后的distance，在-1~+1,out是32*512 epsilon是缩放系数
    Q = torch.exp(out / epsilon).t()  # Q is K-by-B for consistency with notations from our paper  是512*32

    B = Q.shape[1]  #32 代表batch_size
    K = Q.shape[0]  # how many prototypes  类别数，code——number
    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5) #求和
    #全局保持一致
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    #归一化  Q[i, j] 代表样本j归属于簇i的概率
    Q /= sum_Q
    #反复对行和列进行归一化
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.clamp(torch.sum(Q, dim=1, keepdim=True), min=1e-5)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.clamp(torch.sum(torch.sum(Q, dim=0, keepdim=True), dim=1, keepdim=True), min=1e-5)
        Q /= B
    Q *= B  #恢复Q的取值范围
    return Q.t() #进行转置

class Model(nn.Module, GenerationMixin, ABC):
    def __init__(self, model, use_constraint: bool, sk_epsilon: float = 0.03, sk_iters: int = 100, code_length=1,
                 zero_inp=False, code_number=10):
        super().__init__()
        self.model = model #t5
        self.config = model.config 
        self.generation_config = model.generation_config
        self.main_input_name = model.main_input_name
        self.get_encoder = model.get_encoder
        self.device = model.device
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.can_generate = lambda: True
        hidden_size = model.config.hidden_size

        self.use_constraint, self.sk_epsilon, self.sk_iters = use_constraint, sk_epsilon, sk_iters

        # Codebook of each time step
        # code_number为分了多少类，这里是512
        self.centroids = nn.ModuleList([nn.Linear(hidden_size, code_number, bias=False) for _ in range(code_length)])
        self.centroids.requires_grad_(True)

        # Code embedding (input to the decoder)，将类别转换为hidden_size
        self.code_embedding = nn.ModuleList([nn.Embedding(code_number, hidden_size) for _ in range(code_length)])
        self.code_embedding.requires_grad_(True) #计算梯度，可以优化

        self.code_length = code_length
        self.zero_inp = zero_inp
        self.code_number = code_number

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, encoder_outputs=None, **kwargs):
        return {"decoder_input_ids": input_ids, "encoder_outputs": encoder_outputs, "attention_mask": attention_mask}

    @torch.no_grad()
    def quantize(self, probability, use_constraint=None):
        # batchsize_per_device = len(continuous_embeds)
        # distances = ((continuous_embeds.reshape(batchsize_per_device, self.config.MCQ_M, 1, -1).transpose(0,1) -
        #               self.centroids.unsqueeze(1)) ** 2).sum(-1)  # M, bs', K
        distances = -probability #32*512 概率越大，距离越小
        use_constraint = self.use_constraint if use_constraint is None else use_constraint #训练时为True，其他为None
        # raw_code = torch.argmin(distances, dim=-1)
        # print('In', torch.argmin(distances, dim=-1))
        if not use_constraint:
            codes = torch.argmin(distances, dim=-1)  # M, bs
        else:
            distances = self.center_distance_for_constraint(distances)  # to stablize
            # avoid nan
            distances = distances.double() #转换为float64
            # Q = sinkhorn_algorithm(
            #     -distances.transpose(1, 2),
            #     self.sk_epsilon,
            #     self.sk_iters,
            #     use_distrib_train=dist.is_initialized()
            # ).transpose(1, 2)  # M-B-K
            # sk_epsilon=1
            Q = sinkhorn_raw(
                -distances,
                self.sk_epsilon,
                self.sk_iters,
                use_distrib_train=dist.is_initialized()
            )  # B-K
            codes = torch.argmax(Q, dim=-1)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
        # print('Out', codes)
        # print('Equal', (raw_code == codes).float().mean())
        # codes = codes.t()  # bs, M
        # input('>')
        return codes

    def unload(self):
        self.model.unload()

    def decode(self, codes, centroids=None):
        M = codes.shape[1]
        if centroids is None:
            centroids = self.centroids
        if isinstance(codes, torch.Tensor):
            assert isinstance(centroids, torch.Tensor)
            first_indices = torch.arange(M).to(codes.device)
            first_indices = first_indices.expand(*codes.shape).reshape(-1)
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        elif isinstance(codes, np.ndarray):
            if isinstance(centroids, torch.Tensor):
                centroids = centroids.detach().cpu().numpy()
            first_indices = np.arange(M)
            first_indices = np.tile(first_indices, len(codes))
            quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)
        else:
            raise NotImplementedError()
        return quant_embeds

    def embed_decode(self, codes, centroids=None):
        if centroids is None:
            centroids = self.centroids[-1]
        #centroids.weight 是一个(code_number, hidden_size)的矩阵，所以输出是32*768
        quant_embeds = F.embedding(codes, centroids.weight)
        # print(quant_embeds.shape) 32*768
        return quant_embeds

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: M, bs, K
        max_distance = distances.max()
        min_distance = distances.min()
        #如果分布式训练，找到全局的最大值和最小值
        if dist.is_initialized():
            dist.all_reduce(max_distance, torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_distance, torch.distributed.ReduceOp.MIN)
        middle = (max_distance + min_distance) / 2  #找到中间值
        amplitude = max_distance - middle + 1e-5  #最大值到中间值的偏差
        assert torch.all(amplitude > 0)
        centered_distances = (distances - middle) / amplitude  #归一化 max大约为1 min大约为-1
        return centered_distances   #形状不变

    def save_pretrained_t5(self,save_path):
        self.model.model.save_pretrained(save_path)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, aux_ids=None, return_code=False,
                return_quantized_embedding=False, use_constraint=None, encoder_outputs=None, **kwargs):
        if decoder_input_ids is None or self.zero_inp:
            decoder_input_ids = torch.zeros(input_ids.size(0), self.code_length).long().to(input_ids.device)

        # decoder_inputs_embeds = self.code_embedding(decoder_input_ids)

        decoder_inputs_embeds = []
        # 在ids的长度和code_embedding层长度中选个小的
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            code_embedding = self.code_embedding[i] #第i个嵌入层，形状为512，hidden_size
            decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i]))#输入前面的ids到嵌入层里面，得到embeddings然后加到输入embeddings中
        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )
        decoder_outputs = model_outputs.decoder_hidden_states[-1]#获得模型输出的最后一层隐藏状态，(batch_size,sequence_length,hidden_dim) 32*1*768
        all_dense_embed = decoder_outputs.view(decoder_outputs.size(0), -1).contiguous()#(batch_size,sequence_length×hidden_dim)  32*768
        dense_embed = decoder_outputs[:, -1].contiguous()#获取最后一个时间步骤的隐藏状态     32*768
        # print("decoder_outputs.shape",decoder_outputs.shape)
        # print("all_dense_embed.shape",all_dense_embed.shape)
        # print("dense_embed.shape",dense_embed.shape)
        code_logits = []
        for i in range(min(decoder_input_ids.size(1), len(self.code_embedding))):
            #分类层，将embddings转换成不同的类别/簇
            centroid = self.centroids[i]
            code_logits.append(centroid(decoder_outputs[:, i]))
            # print("code_logits",code_logits)
        
        code_logits = torch.stack(code_logits, dim=1)  #32*1*512
        # print("code_logits.shape",code_logits.shape)
        # code_logits = self.centroids(decoder_outputs)

        probability = code_logits[:, -1].contiguous()  #32*512
        # probability = torch.mm(dense_embed, self.centroids.transpose(0, 1))
        discrete_codes = self.quantize(probability, use_constraint=use_constraint)   #进行行归一化和列归一化，针对每个batch选择概率最大的codes

        if aux_ids is None:
            aux_ids = discrete_codes
        # print(aux_ids)
        # print(aux_ids.shape) #32
        quantized_embeds = self.embed_decode(aux_ids) if return_quantized_embedding else None

        if self.code_length == 1:
            return_code_logits = None
        else:
            return_code_logits = code_logits[:, :-1].contiguous() #除了最后一个t以外的所有t

        quant_output = QuantizeOutput(
            logits=code_logits,#32*1*512
            all_dense_embed=all_dense_embed,#32*768  所有的，当没有preid时候与dense一样
            continuous_embeds=dense_embed,#32*768  最后一层的
            quantized_embeds=quantized_embeds,  #将code再转换成embeddings的
            discrete_codes=discrete_codes,  #生成的codes
            probability=probability,  #概率
            code_logits=return_code_logits,
        )
        return quant_output




def get_model(model_name, max_new_tokens=20):
     
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config


def safe_load(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    #获取当前模型的参数key
    model_state_dict_keys = list(model.state_dict().keys())
    #获取要加载的模型参数key
    new_state_dict_keys = list(state_dict.keys())
    #找到要加载模型而不在当前模型中的
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    #相反
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    # size_not_match = [k for k,v in state_dict.items() if model_state_dict_keys[k]]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)
    #只获取匹配的keys
    model.load_state_dict(state_dict, strict=False)



class BiDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_doc_len=512, max_q_len=128, ids=None, batch_size=1, aux_ids=None):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_doc_len = max_doc_len
        self.max_q_len = max_q_len
        self.ids = ids
        self.batch_size = batch_size
        self.aux_ids = aux_ids

    def getitem(self, item):
        queries = self.data[item]
        if isinstance(queries, list):
            query = np.random.choice(queries)
        else:
            query = queries

        doc = self.corpus[item]
        if self.ids is None:
            ids = [0]
        else:
            ids = self.ids[item]
        if self.aux_ids is None:
            aux_ids = -100
        else:
            aux_ids = self.aux_ids[doc_id]
        return (torch.tensor(self.tokenizer.encode(query, truncation=True, max_length=self.max_q_len)),
                torch.tensor(self.tokenizer.encode(doc, truncation=True, max_length=self.max_doc_len)),
                ids, aux_ids)

    def __getitem__(self, item):
        return [self.getitem(item)]
   

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data): 
        data = sum(data, [])
        query, doc, ids, aux_ids = zip(*data)
        # 对序列数据进行填充的，返回(batch_size, max_seq_len, features)，填充短的，query是一个列表，每个元素的形状为(seq_len, features)，但是不是一样的
        query = pad_sequence(query, batch_first=True, padding_value=0)
        doc = pad_sequence(doc, batch_first=True, padding_value=0)
        ids = torch.tensor(ids)
        # 将赋值-100的再转变成None
        if self.aux_ids is None:
            aux_ids = None
        else:
            aux_ids = torch.tensor(aux_ids)
        return {
            'query': query,
            'doc': doc,
            'ids': ids,
            'aux_ids': aux_ids
        }

class OurTrainer:
    @staticmethod
    def _gather_tensor(t: Tensor, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t: Tensor, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(OurTrainer._gather_tensor(t, local_rank))

    @staticmethod
    @torch.no_grad()
    def test_step(model: Model, batch, use_constraint=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=None, return_code=False,
                                              return_quantized_embedding=False, use_constraint=use_constraint)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=None, return_code=False,
                                            return_quantized_embedding=False, use_constraint=use_constraint)
        return query_outputs, doc_outputs

    @staticmethod
    def simple_train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'])
        # doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
        #                                     decoder_input_ids=batch['ids'])

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            code_number = model.module.code_number
        else:
            code_number = model.code_number
        # code_number = 10
        query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number),#每个时间步骤的logits
                                          batch['ids'][:, 1:].reshape(-1)) #去掉t=0时候的token
        # doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, cotrain_steforward
        # de_number), batch['ids'][:, 1:].reshape(-1))
        query_prob = query_outputs.probability
        aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
        code_loss = query_code_loss
        return dict(
            loss=query_code_loss + aux_query_code_loss,
        )

    @staticmethod
    def train_step(model: Model, batch, gathered=None):
        query_outputs: QuantizeOutput = model(input_ids=batch['query'], attention_mask=batch['query'].ne(0),
                                              decoder_input_ids=batch['ids'],
                                              aux_ids=batch['aux_ids'], return_code=True,
                                              return_quantized_embedding=True)
        doc_outputs: QuantizeOutput = model(input_ids=batch['doc'], attention_mask=batch['doc'].ne(0),
                                            decoder_input_ids=batch['ids'],
                                            aux_ids=batch['aux_ids'], return_code=True,
                                            return_quantized_embedding=True)
        query_embeds = query_outputs.continuous_embeds  #最后一层的
        doc_embeds = doc_outputs.continuous_embeds     #最后一层的
        codes_doc = doc_outputs.discrete_codes        #根据文档生成的docid
        quant_doc_embeds = doc_outputs.quantized_embeds   #根据文档生成的docid转换成的embeddings
        query_prob = query_outputs.probability      #根据查询生成的docid概率
        doc_prob = doc_outputs.probability           #根据文档生成的docid的概率

        query_all_embeds = query_outputs.all_dense_embed   #根据查询生成的所有的embeddings
        doc_all_embeds = doc_outputs.all_dense_embed

        if gathered is None:
            gathered = dist.is_initialized()
        #计算此刻根据查询和文档模型输出的对比损失
        cl_loss = OurTrainer.compute_contrastive_loss(query_embeds, doc_embeds, gathered=False)  # retrieval
        #计算所有的查询和文档模型输出的对比损失
        all_cl_loss = OurTrainer.compute_contrastive_loss(query_all_embeds, doc_all_embeds,
                                                          gathered=dist.is_initialized())  # retrieval (used when dist)

        # cl_d_loss = OurTrainer.compute_contrastive_loss(doc_embeds, query_embeds, gathered=gathered)
        # cl_loss = cl_q_loss + cl_d_loss

        # mse_loss = 0
        #计算根据文档code重建生成的embeddings与doc_embeddings的对比损失
        cl_dd_loss = OurTrainer.compute_contrastive_loss(
            quant_doc_embeds + doc_embeds - doc_embeds.detach(), doc_embeds.detach(), gathered=False)  # reconstruction
        # mse_loss = ((quant_doc_embeds - doc_embeds.detach()) ** 2).sum(-1).mean()

        # codes_doc_cpu = codes_doc.cpu().tolist()
        # print(balance(codes_doc_cpu))
        # print(codes_doc)
        #计算交叉熵损失 loss = - ∑ y_true * log(y_pred)
        # ce_loss=0
        # T=query_prob.shape[1]
        # for t in range(T):
        #     ce_loss += F.cross_entropy(query_prob[:, t, :], codes_doc[:, t].detach())  # 计算每个时间步的 loss
        #     ce_loss += F.cross_entropy(doc_prob[:, t, :], codes_doc[:, t].detach())
        # ce_loss = -ce_loss  # 公式中的负号
        print(codes_doc)
        query_ce_loss = F.cross_entropy(query_prob, codes_doc.detach())  # commitment  #32*512   512
        doc_ce_loss = F.cross_entropy(doc_prob, codes_doc.detach())  # commitment
        ce_loss = query_ce_loss + doc_ce_loss  # commitment

        code_loss = 0
        if query_outputs.code_logits is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                code_number = model.module.code_number
            else:
                code_number = model.code_number
            query_code_loss = F.cross_entropy(query_outputs.code_logits.view(-1, code_number), #输出所有的这个code的概率
                                              batch['ids'][:, 1:].reshape(-1))   #去掉t=0时刻的id
            doc_code_loss = F.cross_entropy(doc_outputs.code_logits.view(-1, code_number),
                                            batch['ids'][:, 1:].reshape(-1))
            code_loss = query_code_loss + doc_code_loss  # commitment
        if batch['aux_ids'] is not None:
            aux_query_code_loss = F.cross_entropy(query_prob, batch['aux_ids'])
            aux_doc_code_loss = F.cross_entropy(doc_prob, batch['aux_ids'])
            aux_code_loss = aux_query_code_loss + aux_doc_code_loss  # commitment on last token
            # print('Q', aux_query_code_loss.item(), 'D', aux_doc_code_loss.item())
            if aux_code_loss.isnan():
                aux_code_loss = 0
        else:
            aux_code_loss = 0

        if dist.is_initialized():
            all_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
            global_mean_doc_embeds = all_doc_embeds.mean(dim=0)
            local_mean_doc_embeds = doc_embeds.mean(dim=0)
            clb_loss = F.mse_loss(local_mean_doc_embeds, global_mean_doc_embeds.detach())  # not used
        else:
            clb_loss = 0

        return dict(
            cl_loss=cl_loss,
            all_cl_loss=all_cl_loss,
            mse_loss=0,
            ce_loss=ce_loss,
            code_loss=code_loss,
            aux_code_loss=aux_code_loss,
            cl_dd_loss=cl_dd_loss,
            clb_loss=clb_loss
        )

    @staticmethod
    def compute_contrastive_loss(query_embeds, doc_embeds, gathered=True):
        if gathered:
            gathered_query_embeds = OurTrainer.gather_tensors(query_embeds)
            gathered_doc_embeds = OurTrainer.gather_tensors(doc_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_doc_embeds = doc_embeds
        effective_bsz = gathered_query_embeds.size(0)
        # 每个query匹配的docid的索引
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device) #(batch_size)
        similarities = torch.matmul(gathered_query_embeds, gathered_doc_embeds.transpose(0, 1))  #计算所有的查询和文档的相似性 (batch_size,batch_size)
        # similarities = similarities
        co_loss = F.cross_entropy(similarities, labels)
        #让query[i] 与 doc[i] 之间的相似度最高，远离其他 doc[j]
        return co_loss





def train(config):
    accelerator = Accelerator(gradient_accumulation_steps=1)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_name = config.get('model_name', '/mnt/ljy/model_data/t5-base')
    code_num = config.get('code_num', 512)
    code_length = config.get('code_length', 1)
    prev_model = config.get('prev_model', None)
    prev_id = config.get('prev_id', None)
    save_path = config.get('save_path', None)

    corpus_data = config.get('corpus_data', 'nq320k/corpus_lite.json')
    with open(corpus_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    corpus = [item['doc'] for item in train_data if 'doc' in item]


    epochs = config.get('epochs', 100)
    in_batch_size = config.get('batch_size', 64)
    end_epoch = epochs

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    save_step = 10
    batch_size = 1
    lr = 5e-4


    t5_model, tokenizer, _generation_config = get_model(model_name)
    checkpoint_name = config.get('checkpoint', None)
    print('Loading checkpoint:', checkpoint_name)
    safe_load(t5_model, checkpoint_name)  #加载模型
    ROOT_DIR ='/mnt/ljy/GRR/data_nq320k/prag'
    init_adapter_path = os.path.join(
        ROOT_DIR,
        "offline",
        "t5",
        f"rank={config['lora_rank']}_alpha={config['lora_alpha']}",
        "base_weight",
    )
    if not os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors")):
        print("No LoRA base weight, creating...")
        peft_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q", "v"],
            inference_mode=False,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=0, # !!!
        )
        t5_model = get_peft_model(t5_model, peft_config)
        
        t5_model.is_parallelizable = True
        t5_model.model_parallel = True
        print(f'Save LoRA base weight to {init_adapter_path}')
        os.makedirs(init_adapter_path, exist_ok=True)
        t5_model.save_pretrained(init_adapter_path)
        time.sleep(2)
        assert os.path.exists(os.path.join(init_adapter_path, "adapter_model.safetensors"))
    

    t5_model = PeftModel.from_pretrained(t5_model, init_adapter_path, is_trainable=True)
    t5_model.is_parallelizable = True
    t5_model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, t5_model.parameters()) #筛选出需要梯度更新的参数
    model = Model(model=t5_model, code_length=code_length,
                  use_constraint=True, sk_epsilon=1, zero_inp=False, code_number=code_num)
    
    if prev_model is not None:
        safe_load(model.model, f'{prev_model}.model')
        safe_load(model.centroids, f'{prev_model}.centroids')
        safe_load_embedding(model.code_embedding, f'{prev_model}.embedding')

    if config.get('codebook_init', None) is not None:
        model.centroids[-1].weight.data = torch.tensor(read_pkl(config.get('codebook_init')))
    #只更新当前的质心，之前的都不更新
    for i in range(code_length - 1):
        model.centroids[i].requires_grad_(False)


    grouped_data = []  #创建一个字典类defaultdict(<class 'list'>, {'fruit': ['apple', 'banana'], 'vegetable': ['carrot', 'spinach']})
    for i, item in enumerate(train_data):
        query= item['queries']
        grouped_data.append(query) #对应字典，将相同docid的查询放在一起
      #[['is airplane fuel the same as car fuel', 'what is name of fuel used in aeroplane', "what's the difference between gas and jet fuel"], ['all that remains what if i was nothing album']]


    if prev_id is not None:
        ids = [[0, *line] for line in json.load(open(prev_id))]
    else:
        ids = [[0]] * len(corpus)
    aux_ids = None


    for idx,item in enumerate(corpus):
        only_grouped_data = grouped_data[idx:idx+1]
        only_corpus = corpus[idx:idx+1]
        only_ids = ids[idx:idx+1]
        dataset = BiDataset(data=only_grouped_data, corpus=only_corpus, tokenizer=tokenizer,
                            max_doc_len=128, max_q_len=32, ids=only_ids, batch_size=1, aux_ids=aux_ids)

        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=1,
                                                shuffle=True, num_workers=1)


        optimizer = AdamW(model.parameters(), lr)
        model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
        scheduler = get_constant_schedule(optimizer)

        w_1 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
            'cl_dd_loss': 0, 'clb_loss': 0}
        w_2 = {'cl_loss': 0.5, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 0.5, 'aux_code_loss': 0, 'mse_loss': 0,
            'cl_dd_loss': 0, 'clb_loss': 0}
        w_3 = {'cl_loss': 0, 'all_cl_loss': 0, 'ce_loss': 0.5, 'code_loss': 1, 'aux_code_loss': 0, 'mse_loss': 0,
            'cl_dd_loss': 0, 'clb_loss': 0}
        loss_w = [None, w_1, w_2, w_3][config['loss_w']]

        step, epoch = 0, 0
        epoch_step = len(data_loader)
        # safe_save(accelerator, model, save_path, -1, end_epoch=end_epoch)
        last_checkpoint = None

        for _ in range(epochs):
            accelerator.wait_for_everyone()
            model.train()
            tk0 = tqdm(data_loader, total=len(data_loader))
            loss_report = []
            for batch in tk0:
                step += 1
                # 可以累计多个梯度
                with accelerator.accumulate(model):
                    losses = OurTrainer.train_step(model, batch, gathered=False)
                    print(losses)
                    loss = sum([v * loss_w[k] for k, v in losses.items()])
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    loss = accelerator.gather(loss).mean().item()
                    loss_report.append(loss)
                    tk0.set_postfix(loss=sum(loss_report[-100:]) / len(loss_report[-100:]))

                    save_path_passage = f"{save_path}/passage_{idx}"
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained_t5(save_path_passage)
                    # model = model.unload()
                    torch.cuda.empty_cache()
                    gc.collect()
                # epoch = safe_save(accelerator, model, save_path, epoch, end_epoch=end_epoch, save_step=save_step)
    return last_checkpoint

def safe_load_embedding(model, file):
    state_dict = torch.load(file, map_location=lambda storage, loc: storage)
    model_state_dict_keys = list(model.state_dict().keys())
    new_state_dict_keys = list(state_dict.keys())
    new_keys_in_new = [k for k in new_state_dict_keys if k not in model_state_dict_keys]
    no_match_keys_of_model = [k for k in model_state_dict_keys if k not in new_state_dict_keys]
    print('##', model._get_name(), '# new keys in file:', new_keys_in_new, '# no match keys:', no_match_keys_of_model)

    matched_state_dict = deepcopy(model.state_dict())
    for key in model_state_dict_keys:
        if key in state_dict:
            file_size = state_dict[key].size(0)
            model_embedding = matched_state_dict[key].clone()
            model_size = model_embedding.size(0)
            model_embedding[:file_size, :] = state_dict[key][:model_size, :]
            matched_state_dict[key] = model_embedding
            print(f'Copy {key} {matched_state_dict[key].size()} from {state_dict[key].size()}')
    model.load_state_dict(matched_state_dict, strict=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/mnt/ljy/model_data/t5-base')   #模型名称
    parser.add_argument('--code_num', type=int, default=512)       #编码数量
    parser.add_argument('--max_length', type=int, default=3)     
    parser.add_argument('--corpus_data', type=str, default='/mnt/ljy/GRR/data_nq320k/updat_sdoc_to_queries.json')   #{'doc_id': 98026, 'doc': "Strom state to state arele by 2030 .", 'queries': ['upper class middle class lower class in india']}
    parser.add_argument('--save_path', type=str, default='out1/model-2/corpuslora') #保存路径
    parser.add_argument('--checkpoint', type=str, default='/mnt/ljy/GRR/out1/model-3/20.pt')
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16) 
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def main():
    args = parse_args()   #解析参数
    config = copy.deepcopy(vars(args))#深拷贝参数

    checkpoint = '/mnt/ljy/GRR/out1/model-2/10.pt'

    for loop in range(2,args.max_length):
        config['save_path'] = args.save_path + f'-{loop}'
        config['code_length'] = loop + 1
        config['prev_model'] = checkpoint
        config['prev_id'] = f'{checkpoint}.update.code' if checkpoint is not None else None
        config['epochs'] = 1
        config['loss_w'] = 1
        config['checkpoint'] = args.checkpoint
        config['code_num'] = args.code_num
        config['lora_rank'] = args.lora_rank
        config['lora_alpha'] = args.lora_alpha
        train(config)

if __name__ == '__main__':
    main()



