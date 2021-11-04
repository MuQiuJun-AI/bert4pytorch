#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4pytorch.modeling import build_transformer_model
from bert4pytorch.tokenization import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "D:/vscodeworkspace/pythonCode/my_project/my_pytorch_bert/pytorch_bert_pretrain_model"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path)
sentence = "北京[MASK]安门"


tokens_ids, segments_ids = tokenizer.encode(sentence)
mask_position = tokens_ids.index(103)

tokens_ids_tensor = torch.tensor([tokens_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model = build_transformer_model(config_path, checkpoint_path, with_mlm=True)
model.eval()
output = model(tokens_ids_tensor, segment_ids_tensor)

result = torch.argmax(output[0, mask_position]).item()

print(tokenizer.convert_ids_to_tokens([result])[0])
