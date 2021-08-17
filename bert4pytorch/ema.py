class EMA():
    '''
        模型权重的指数滑动平均
        注意区别于类似adam一类的自适应学习率优化器，针对一阶二阶梯度的指数滑动平均，两者完全不同

        例子:
            # 初始化
            ema = EMA(model, 0.999)

            # 训练过程中，更新完参数后，同步update ema_weights weights
            def train():
                optimizer.step()
                ema.update()

            # eval前，apply ema_weights weights；eval之后，恢复原来模型的参数
            def evaluate():
                ema.apply_ema_weights()
                # evaluate
                # 如果想保存ema后的模型，请在restore方法之前调用torch.save()
                ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        # 保存影子权重（当前step的每一层的滑动平均权重）
        self.ema_weights = {}
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从影子权重恢复到原始权重
        self.model_weights = {}

        # 初始化ema_weights为model_weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                new_average = (1.0 - self.decay) * param.data + self.decay * self.ema_weights[name]
                self.ema_weights[name] = new_average.clone()
    
    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                self.model_weights[name] = param.data
                param.data = self.ema_weights[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.model_weights
                param.data = self.model_weights[name]
        self.model_weights = {}




