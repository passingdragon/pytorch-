import torch
from torch import nn
from torch.nn import functional as F

# 自己实现一些块
X = torch.rand(2,20)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
net = MLP()
print(net(X))


class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self._module = {}
        for idx,module in enumerate(args):
            self._module[str(idx)] = module
    def forward(self,X):
        for block in self._module.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(X))



class FixedHidMLP(nn.Module):
    def __init__(self):
        super().__init__()
        #一个永远都不会更新的权重
        self.rand_wight = torch.rand((20,20),requires_grad=False)
        self.liner = nn.Linear(20,20)
    def forward(self,X):
        X = self.liner(X)
        # 一个永远的不会更新的隐藏层
        X = F.relu(torch.mm(X,self.rand_wight)+1)
        X = self.liner(X)
        return X


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.liner = nn.Linear(32,10)
    def forward(self,X):
        return self.liner(self.net(X))
mynet = nn.Sequential(NestMLP(),nn.Linear(10,20),FixedHidMLP(),MLP())
print(mynet(X))