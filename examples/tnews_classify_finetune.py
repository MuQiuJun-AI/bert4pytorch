from torch.utils.data import Dataset, DataLoader
from bert4pytorch.modeling_new import build_transformer_model
from bert4pytorch.tokenization import Tokenizer
from bert4pytorch.optimization import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import json
import time

SEED = 100
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

learning_rate = 2e-5
epochs = 50
max_len = 32
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() > 0 else "cpu")


# 加载模型，请更换成自己的路径
root_model_path = "D:/vscodeworkspace/pythonCode/my_project/my_pytorch_bert/pytorch_bert_pretrain_model"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    texts = []
    labels = []
    with open(filename, encoding='utf8') as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text = l['sentence']
            label = l['label']
            texts.append(text)
            label_int = int(label)
            # 数据集标签转成pytorch需要的形式[0, 类别数-1]
            if label_int <= 104:
                labels.append(label_int - 100)
            elif 104 < label_int <= 110:
                labels.append(label_int - 101)
            else:
                labels.append(label_int - 102)
    return texts, labels


# 加载数据集，请更换成自己的路径
X_train, y_train = load_data('D:/vscodeworkspace/pythonCode/git_workspace/clue_dataset/tnews_public/train.json')
X_test, y_test = load_data('D:/vscodeworkspace/pythonCode/git_workspace/clue_dataset/tnews_public/dev.json')


# 建立分词器
tokenizer = Tokenizer(vocab_path)


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sentence = self.X[index]
        label = self.y[index]
        tokens_ids, segments_ids = tokenizer.encode(sentence, max_len=max_len)
        tokens_ids = tokens_ids + (max_len - len(tokens_ids)) * [0]
        segments_ids = segments_ids + (max_len - len(segments_ids)) * [0]
        tokens_ids_tensor = torch.tensor(tokens_ids)
        segment_ids_tensor = torch.tensor(segments_ids)
        return tokens_ids_tensor, segment_ids_tensor, label


class Model(nn.Module):

    def __init__(self, config, checkpoint):
        super(Model, self).__init__()
        self.model = build_transformer_model(config, checkpoint, with_pool=True)
        '''所有层都训练'''
        for param in self.model.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(768, 15)

    def forward(self, token_ids, segment_ids):
        encoded_layers, pooled_output = self.model(token_ids, segment_ids)
        # 取最后一个输出层的第一个位置
        cls_rep = self.dropout(encoded_layers[:, 0])
        out = self.fc(cls_rep)
        return out

# 构建dataset
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
#构建dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 模型实例化
model = Model(config_path, checkpoint_path).to(device)
# 定义损失函数
critertion = nn.CrossEntropyLoss()
# 权重衰减，layernorn层。以及每一层的bias不进行权重衰减
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'layerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# 使用warmup
num_training_steps = (len(train_dataloader) + 1) * epochs
num_warmup_steps = num_training_steps * 0.05
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

total_step = len(train_dataloader)
loss_list = []
train_non_ema_acc_list = []
train_ema_acc_list = []
test_non_ema_acc_list = []
test_ema_acc_list = []

best_acc = 0.0
model.train()
for epoch in range(epochs):
    start = time.time()
    for i, (token_ids, segment_ids, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        labels = labels.to(device)
        outputs = model(token_ids, segment_ids)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())

        if (i % 100) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, spend time: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item(), time.time() - start))
            start = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (token_ids, segment_ids, labels) in enumerate(test_dataloader):
            token_ids = token_ids.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)
            outputs = model(token_ids, segment_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_non_ema_acc = correct / total
        train_non_ema_acc_list.append(train_non_ema_acc)

        print('non ema Epoch [{}/{}], train_non_ema_acc: {:.6f}'
              .format(epoch + 1, epochs, train_non_ema_acc))
    model.train()
