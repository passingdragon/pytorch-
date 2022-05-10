import torch
from torch import nn
# library为自己写的库，随着深度学习进度内容会不断增加
import library
from matplotlib import pyplot as plt
# 加载数据集
batch_size = 256
train_iter,test_iter = library.load_data_fashion(batch_size)

num_in = 28*28
num_out = 10

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
# 初始化线性层参数
def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
# 将初始化参数放入网络中
net.apply(init_weight)
# 定义损失函数，此处为交叉熵函数
loss = nn.CrossEntropyLoss()
# 优化器
trainer = torch.optim.SGD(net.parameters(),lr = 0.01)
# 训练次数


num_epochs = 10
# 记录损失，用来可视化
train_loss = []
for i in range(num_epochs):
    for index,(X,y) in enumerate(train_iter):
        # 向前传播
        y_ = net(X)
        # 单热点编码
        y_onehot = F.one_hot(y, num_classes=10).float()
        l = loss(y_,y_onehot).sum()
        # 梯度清零
        trainer.zero_grad()
        # 向后传播
        l.backward()
        # 更新
        trainer.step()
        train_loss.append(l.item())
        if index%100 == 0:
            print(i,l.item())

library.pltLoss(train_loss)
print(library.evaluate_accuracy(net,test_iter))
# 0.7703


