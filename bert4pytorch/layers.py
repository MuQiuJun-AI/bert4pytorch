import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def gelu(x):
    """ gelu激活函数
        在GPT架构中，使用的是gelu函数的近似版本，公式如下:
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        这里是直接求的解析解，就是原始论文给出的公式
        论文 https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


activations = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.conditional = conditional
        if conditional:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.eps)
            return (self.weight + self.dense1(cond)) * x + (self.bias + self.dense2(cond))
        else:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_scale=True,
                 return_attention_scores=False):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None):

        # query shape: [batch_size, query_len, hidden_size]
        # key shape: [batch_size, key_len, hidden_size]
        # value shape: [batch_size, value_len, hidden_size]
        # 一般情况下，query_len、key_len、value_len三者相等

        mixed_query_layer = self.q(query)
        mixed_key_layer = self.k(key)
        mixed_value_layer = self.v(value)

        # mixed_query_layer shape: [batch_size, query_len, hidden_size]
        # mixed_query_layer shape: [batch_size, key_len, hidden_size]
        # mixed_query_layer shape: [batch_size, value_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]

        # 是否进行attention scale
        if self.attention_scale:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分
        if attention_mask is not None:
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]

        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 所以在调用view之前，需要contiguous来返回一个contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.5, hidden_act='gelu', is_dropout=True):
        # 原生的tf版本的bert在激活函数后，没有添加dropout层，但是在google AI的bert-pytorch开源项目中，多了一层dropout；
        # 并且在pytorch官方的TransformerEncoderLayer的实现中，也有一层dropout层，就像这样：self.linear2(self.dropout(self.activation(self.linear1(src))))；
        # 这样不统一做法的原因不得而知，不过有没有这一层，差别可能不会很大；

        # 为了适配是否dropout，用is_dropout，dropout_rate两个参数控制；如果是实现原始的transformer，直接使用默认参数即可；如果是实现bert，则is_dropout为False，此时的dropout_rate参数并不会使用.
        super(PositionWiseFeedForward, self).__init__()

        self.is_dropout = is_dropout
        self.intermediate_act_fn = activations[hidden_act]
        self.intermediateDense = nn.Linear(hidden_size, intermediate_size)
        self.outputDense = nn.Linear(intermediate_size, hidden_size)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch size, seq len, hidden_size)
        if self.is_dropout:
            x = self.dropout(self.intermediate_act_fn(self.intermediateDense(x)))
        else:
            x = self.intermediate_act_fn(self.intermediateDense(x))

        # x shape: (batch size, seq len, intermediate_size)
        x = self.outputDense(x)

        # x shape: (batch size, seq len, hidden_size)
        return x
