# bert4pytorch

## 更新：

- **2021年8月27更新**：感谢大家的star，最近有小伙伴反映了一些小的bug，我也注意到了，奈何这个月工作上实在太忙，更新不及时，大约会在9月中旬集中更新一个只需要pip一下就完全可用的版本，然后会新添加一些关键注释。
再增加对抗训练的内容，更新一个完整的finetune案例。
- **2021年9月6日更新**：<br>
  1、删除file_utils文件, 简化加载预训练模型代码和网络请求库的依赖, 这样就只支持下载相关模型文件后，本地加载模型，模型可以去这里下载：https://huggingface.co/models<br>
  2、增加特殊的layers、特殊的loss, layer增加了CRF，loss增加了focal_loss和LabelSmoothingCrossEntropy, 后续会逐步添加<br>
- **2021年11月3日更新**：<br>
  考虑到后续对bert家族，比如albert、T5、NEZHA、ELECTRA等架构的实现能全部集中在一个model文件实现，保证代码简洁清爽，本次更新基本对代码进行了全面重构，主干参照了bert4keras的代码结构。几乎可以以bert4keras的api风格使用。另外实现了unilm式、gpt式的mask矩阵。使用例子后续会给出。
  其他几点更新如下：<br>
  1、删除ema文件，把权重滑动平均整合到optimization文件<br>
  2、添加一个完整的分类案例，在CLUE的tnews数据集上做finetune<br>
- **2021年11月4日更新**： 基础测试, 添加mlm预测案例
- **2022年3月22日更新**： focal loss更新，并在分类任务上测试通过
- **2022年3月24日更新**:  实现对抗训练（FGM），并在分类任务上测试通过
  
  

# 背景

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

- albert、GPT、XLnet、conformer等网络架构
- 实现各种trick（比如对抗训练、ema等），定义特定的layer、loss，方便后续扩展
- 添加大量nlp、语音识别的完整可直接运行的例子和中文注释，减轻学习难度


## 使用

##### pip安装
```python
pip install bert4pytorch==0.1.3
```
目前pip安装的是旧版本，新版本请自行下载源码安装

#### 下载源码安装
```python
pip install git+https://github.com/MuQiuJun-AI/bert4pytorch.git
```

## 权重
支持加载的权重

- Google原版bert的pytorch版本(需要转换为pytorch版本的脚本): https://github.com/google-research/bert
- 哈工大版roberta: https://github.com/ymcui/Chinese-BERT-wwm


## 其他

最初整理这个项目，只是为了自己方便。这一段时间，经常逛苏剑林大佬的博客，里面的内容写得相当精辟，更加感叹的是， 苏神经常能闭门造车出一些还不错的trick，只能说，大佬牛逼。

所以本项目命名也雷同bert4keras，以感谢苏大佬无私的分享。

后来，慢慢萌生把学习中的小小成果开源出来，后期会渐渐补充例子，前期会借用苏神的bert4keras里面的例子，实现pytorch版本。如果有问题，欢迎讨论；如果本项目对您有用，请不吝star！
