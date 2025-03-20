# tsrl
## 2025.3.19
完成了利用重写任务，根据文档重写文档，然后利用一个嵌入模型，分别对生成的文档内容和原始的文档进行编码，然后计算相似性作为奖励规则，利用GRPO训练
## 2025.3.20
不使用嵌入模型，使用训练模型得到嵌入，首先使用分词器tokenizer得到input_ids，然后通过 model.get_input_embeddings()得到所有的token embeddings 形状为(batch_size,seq_len,dim)，然后计算平均池化，得到句子级别的嵌入，计算相似性，观察图像，奖励不正确，发现了问题，使用Qwen2-0.5B-Instuct给出的是答案，而且一旦不给问题，给文章，就根本不会生成符合标准的内容，所以决定换个模型来看，换成查询重写zstanjj/SlimPLM-Query-Rewriting试试这个怎么样。
