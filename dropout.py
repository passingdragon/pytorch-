import torch
from torch import nn
import library
from torch.nn import functional as F

batch_size = 256
train_iter, test_iter = library.load_data_fashion(batch_size=batch_size)
input = 28 * 28
output = 10
mid = 256
dropout1 = 0.2
dropout2 = 0.3
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(input, mid), nn.ReLU(), nn.Dropout(dropout1),
                    nn.Linear(mid, mid), nn.ReLU(), nn.Dropout(dropout2),
                    nn.Linear(mid, output))


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


net.apply(init_weight)
# 交叉熵函数

loss = nn.CrossEntropyLoss()
# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.3)

num_epochs = 10
train_loss = []
for epoch in range(num_epochs):
    for index, (X, y) in enumerate(train_iter):
        y_onehot = F.one_hot(y, num_classes=10).float()
        y_pre = net(X)
        l = loss(y_pre, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        train_loss.append(l.item())
        if index % 20 == 0:
            print(l.item())
library.pltLoss(train_loss)
print(library.evaluate_accuracy(net, test_iter))
# 0.8691
