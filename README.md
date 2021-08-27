# bert4pytorch

## 2021年8月27更新：
感谢大家的star，最近有小伙伴反映了一些小的bug，我也注意到了，奈何这个月工作上实在太忙，更新不及时，大约会在9月中旬集中更新一个只需要pip一下就完全可用的版本，然后会新添加一些关键注释。
再增加对抗训练的内容，更新一个完整的finetune案例。

目前最流行的pytorch版本的bert框架，莫过于huggingface团队的Transformers项目，但是随着项目的越来越大，显得很重，对于初学者、有一定nlp基础的人来说，想看懂里面的代码逻辑，深入了解bert，有很大的难度。

另外，如果想修改Transformers的底层代码也是想当困难的，导致很难对模型进行魔改。

本项目把整个bert架构，**浓缩在几个文件当中**（主要修改自Transfomers开源项目），删除大量无关紧要的代码，新增了一些功能，比如：ema、warmup schedule，并且在核心部分，**添加了大量中文注释**，力求解答读者在使用过程中产生的一些疑惑。

此项目核心只有三个文件，modeling、tokenization、optimization。并且都在**几百行内完成**。结合大量的中文注释，分分钟透彻理解bert。

## 功能

## 现在已经实现

- 加载bert、RoBERTa-wwm-ext的预训练权重进行fintune
- 实现了带warmup的优化器
- 实现了模型权重的指数滑动平均（ema）

### 未来将实现

- albert、GPT、XLnet等网络架构
- 实现对抗训练、conditional Layer Norm等功能（想法来自于苏神(苏剑林)的bert4keras开源项目，事实上，bert4pytorch就是受到了它的启发）
- 添加大量的例子和中文注释，减轻学习难度

## 安装

```python
pip install bert4pytorch==0.1.2
```

## 使用

- 加载预训练模型

```python
from bert4pytorch.modeling import BertModel, BertConfig
from bert4pytorch.tokenization import BertTokenizer
from bert4pytorch.optimization import AdamW, get_linear_schedule_with_warmup
import torch

model_path = "/model/pytorch_bert_pretrain_model"
config = BertConfig(model_path + "/config.json")

tokenizer = BertTokenizer(model_path + "/vocab.txt")
model = BertModel.from_pretrained(model_path, config)

input_ids, token_type_ids = tokenizer.encode("今天很开心")

input_ids = torch.tensor([input_ids])
token_type_ids = torch.tensor([token_type_ids])

model.eval()

outputs = model(input_ids, token_type_ids, output_all_encoded_layers=True)

## orther code
```

- 带warmup的优化器实现

```python
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

num_warmup_steps=num_training_steps * warmup_proportion
num_training_steps=train_batches * num_epoches
schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
```

## 其他

最初整理这个项目，只是为了自己方便。这一段时间，经常逛苏剑林大佬的博客，里面的内容写得相当精辟，更加感叹的是， 苏神经常能闭门造车出一些还不错的trick，只能说，大佬牛逼。

所以本项目命名也雷同bert4keras，以感谢苏大佬无私的分享。

后来，慢慢萌生把学习中的小小成果开源出来，后期会渐渐补充例子，前期会借用苏神的bert4keras里面的例子，实现pytorch版本。如果有问题，欢迎讨论；如果本项目对您有用，请不吝star！
