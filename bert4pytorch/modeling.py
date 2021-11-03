import torch
import torch.nn as nn
import copy
import json
from bert4pytorch.layers_new import LayerNorm, MultiHeadAttentionLayer, PositionWiseFeedForward, activations


class Transformer(nn.Module):
    """模型基类
    """

    def __init__(
            self,
            vocab_size,  # 词表大小
            hidden_size,  # 编码维度
            num_hidden_layers,  # Transformer总层数
            num_attention_heads,  # Attention的头数
            intermediate_size,  # FeedForward的隐层维度
            hidden_act,  # FeedForward隐层的激活函数
            dropout_rate,  # Dropout比例
            embedding_size=None,  # 指定embedding_size, 不指定则使用config文件的参数
            attention_head_size=None,  # Attention中V的head_size
            attention_key_size=None,  # Attention中Q,K的head_size
            sequence_length=None,  # 是否固定序列长度
            keep_tokens=None,  # 要保留的词ID列表
            compound_tokens=None,  # 扩展Embedding
            residual_attention_scores=False,  # Attention矩阵加残差
            ignore_invalid_weights=False,  # 允许跳过不存在的权重
            **kwargs
    ):
        super(Transformer, self).__init__()
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or self.hidden_size // self.num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights

    def init_model_weights(self, module):
        raise NotImplementedError

    def variable_mapping(self):
        """构建pytorch层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_pytorch_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        # model = self
        state_dict = torch.load(checkpoint, map_location='cpu')
        mapping = mapping or self.variable_mapping()

        for new_key, old_key in mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict, strict=self.ignore_invalid_weights)


def lm_mask(segment_ids):
    """定义下三角Attention Mask（语言模型用）
    """
    idxs = torch.arange(0, segment_ids.shape[1])
    mask = (idxs.unsqueeze(0) <= idxs.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    return mask


def unilm_mask(segment_ids):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
        其中source和target的分区，由segment_ids来表示。
        UniLM: https://arxiv.org/abs/1905.03197
    """

    # 在序列维度进行累加求和
    idxs = torch.cumsum(segment_ids, dim=1)
    # 构造unilm的mask矩阵，并把shape扩充到[batch_size, num_heads, from_seq_length, to_seq_length]
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).unsqueeze(1).to(dtype=torch.float32)
    return mask


####################################################################################
#       bert                                                                       #
####################################################################################


class BertEmbeddings(nn.Module):
    """
        embeddings层
        构造word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position, segment_vocab_size, drop_rate):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.segment_embeddings = nn.Embedding(segment_vocab_size, hidden_size)

        self.layerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, token_ids, segment_ids=None):
        seq_length = token_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)

        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
        Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm

        注意: 1、以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
              2、原始的Transformer的encoder中的Feed Forward层一共有两层linear，
              config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, intermediate_size, hidden_act, is_dropout=False):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(hidden_size, num_attention_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(hidden_size, eps=1e-12)
        self.feedForward = PositionWiseFeedForward(hidden_size, intermediate_size, hidden_act, is_dropout=is_dropout)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask):
        self_attn_output = self.multiHeadAttention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        return hidden_states


class BERT(Transformer):
    """构建BERT模型
    """

    def __init__(
            self,
            max_position,  # 序列最大长度
            segment_vocab_size=2,  # segment总数目
            initializer_range=0.02, # 权重初始化方差
            with_pool=False,  # 是否包含Pool部分
            with_nsp=False,  # 是否包含NSP部分
            with_mlm=False,  # 是否包含MLM部分
            hierarchical_position=None,  # 是否层次分解位置编码
            custom_position_ids=False,  # 是否自行传入位置id
            **kwargs
    ):
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.initializer_range = initializer_range
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

        super(BERT, self).__init__(**kwargs)

        self.embeddings = BertEmbeddings(self.vocab_size, self.hidden_size, self.max_position, self.segment_vocab_size, self.dropout_rate)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.intermediate_size, self.hidden_act, is_dropout=False)
        self.encoderLayer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)])
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
            self.pooler_activation = nn.Tanh()
            if self.with_nsp:
                # Next Sentence Prediction部分
                # nsp的输入为pooled_output, 所以with_pool为True是使用nsp的前提条件
                self.nsp = nn.Linear(self.hidden_size, 2)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDecoder = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDense = nn.Linear(self.hidden_size, self.hidden_size)
            self.transform_act_fn = activations[self.hidden_act]
            self.mlmLayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.apply(self.init_model_weights)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正太分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, token_ids, segment_ids=None, attention_mask=None, output_all_encoded_layers=False):
        """
            token_ids： 一连串token在vocab中对应的id
            segment_ids： 就是token对应的句子id,值为0或1（0表示对应的token属于第一句，1表示属于第二句）,当
                             任务只有一个句子输入时，segment_ids的每个值都是0，可不用传值
            attention_mask：各元素的值为0或1,避免在padding的token上计算attention, 1进行attetion, 0不进行attention

            以上三个参数的shape为： (batch_size, sequence_length); type为tensor
        """

        if attention_mask is None:
            # 根据token_ids创建一个3D的attention mask矩阵，尺寸为[batch_size, 1, 1, to_seq_length]，
            # 目的是为了适配多头注意力机制，从而能广播到[batch_size, num_heads, from_seq_length, to_seq_length]尺寸
            attention_mask = (token_ids != 0).long().unsqueeze(1).unsqueeze(2)
        if segment_ids is None:
            segment_ids = torch.zeros_like(token_ids)

        # 兼容fp16
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        # 对mask矩阵中，数值为0的转换成很大的负数，使得不需要attention的位置经过softmax后,分数趋近于0
        # attention_mask = (1.0 - attention_mask) * -10000.0
        # 执行embedding
        hidden_states = self.embeddings(token_ids, segment_ids)
        # 执行encoder
        encoded_layers = [hidden_states] # 添加embedding的输出
        for layer_module in self.encoderLayer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not output_all_encoded_layers:
            encoded_layers.append(hidden_states)

        # 获取最后一层隐藏层的输出
        sequence_output = encoded_layers[-1]
        # 是否取最后一层输出
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加pool层
        if self.with_pool:
            pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))
        else:
            pooled_output = None
        # 是否添加nsp
        if self.with_pool and self.with_nsp:
            nsp_scores = self.nsp(pooled_output)
        else:
            nsp_scores = None
        # 是否添加mlm
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_scores = self.mlmDecoder(mlm_hidden_state) + self.mlmBias
        else:
            mlm_scores = None
        # 根据情况返回值
        if mlm_scores is None and nsp_scores is None:
            return encoded_layers, pooled_output
        elif mlm_scores is not None and nsp_scores is not None:
            return mlm_scores, nsp_scores
        elif mlm_scores is not None:
            return mlm_scores
        else:
            return nsp_scores

    def variable_mapping(self):
        mapping = {
            'embeddings.word_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'embeddings.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'embeddings.segment_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'embeddings.layerNorm.weight': 'bert.embeddings.LayerNorm.gamma',
            'embeddings.layerNorm.bias': 'bert.embeddings.LayerNorm.beta',
            'pooler.weight': 'bert.pooler.dense.weight',
            'pooler.bias': 'bert.pooler.dense.bias',
            'nsp.weight': 'cls.seq_relationship.weight',
            'nsp.bias': 'cls.seq_relationship.bias',
            'mlmDense.weight': 'cls.predictions.transform.dense.weight',
            'mlmDense.bias': 'cls.predictions.transform.dense.bias',
            'mlmLayerNorm.weight': 'cls.predictions.transform.LayerNorm.gamma',
            'mlmLayerNorm.bias': 'cls.predictions.transform.LayerNorm.beta',
            'mlmBias': 'cls.predictions.bias',
            'mlmDecoder.weight': 'cls.predictions.decoder.weight'

        }
        for i in range(self.num_hidden_layers):
            prefix = 'bert.encoder.layer.%d.' % i
            mapping.update({'encoderLayer.%d.multiHeadAttention.q.weight' % i: prefix + 'attention.self.query.weight',
                            'encoderLayer.%d.multiHeadAttention.q.bias' % i: prefix + 'attention.self.query.bias',
                            'encoderLayer.%d.multiHeadAttention.k.weight' % i: prefix + 'attention.self.key.weight',
                            'encoderLayer.%d.multiHeadAttention.k.bias' % i: prefix + 'attention.self.key.bias',
                            'encoderLayer.%d.multiHeadAttention.v.weight' % i: prefix + 'attention.self.value.weight',
                            'encoderLayer.%d.multiHeadAttention.v.bias' % i: prefix + 'attention.self.value.bias',
                            'encoderLayer.%d.multiHeadAttention.o.weight' % i: prefix + 'attention.output.dense.weight',
                            'encoderLayer.%d.multiHeadAttention.o.bias' % i: prefix + 'attention.output.dense.bias',
                            'encoderLayer.%d.layerNorm1.weight' % i: prefix + 'attention.output.LayerNorm.gamma',
                            'encoderLayer.%d.layerNorm1.bias' % i: prefix + 'attention.output.LayerNorm.beta',
                            'encoderLayer.%d.feedForward.intermediateDense.weight' % i: prefix + 'intermediate.dense.weight',
                            'encoderLayer.%d.feedForward.intermediateDense.bias' % i: prefix + 'intermediate.dense.bias',
                            'encoderLayer.%d.feedForward.outputDense.weight' % i: prefix + 'output.dense.weight',
                            'encoderLayer.%d.feedForward.outputDense.bias' % i: prefix + 'output.dense.bias',
                            'encoderLayer.%d.layerNorm2.weight' % i: prefix + 'output.LayerNorm.gamma',
                            'encoderLayer.%d.layerNorm2.bias' % i: prefix + 'output.LayerNorm.beta'
                            })

        return mapping


def build_transformer_model(
        config_path=None,
        checkpoint_path=None,
        model='bert',
        application='encoder',
        **kwargs
):
    """根据配置文件构建模型，可选加载checkpoint权重
    """
    configs = {}
    if config_path is not None:
        configs.update(json.load(open(config_path)))
    configs.update(kwargs)
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
    models = {
        'bert': BERT,
        'roberta': BERT
    }

    my_model = models[model]
    transformer = my_model(**configs)
    if checkpoint_path is not None:
        transformer.load_weights_from_pytorch_checkpoint(checkpoint_path)
    return transformer
