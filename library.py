import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

def create_data(w, b, num_data):
    X = torch.normal(0, 1, (num_data, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_data_fashion(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # resize改变图像维度，1*28*28-》1*resize*resize
    trans = transforms.Compose(trans)
    trains = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    tests = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    return (data.DataLoader(trains, batch_size, shuffle=True, num_workers=0),
            data.DataLoader(tests, batch_size, shuffle=True, num_workers=0))
# trains_iter,tests_iter = load_data_fashion(32)
# for x,y in tests_iter:
#     print(x.shape)


#绘制损失曲线
def pltLoss(train_loss:list):
    plt.figure(figsize=(16,8))
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.plot(train_loss,label='loss')
    plt.legend(loc="best")
    plt.show()
def sumAccuracy(y_pre,y):
    # 预测正确的数量
    if len(y_pre.shape)>1 and y_pre.shape[1] > 1:
        y_pre = y_pre.argmax(axis = 1)
    cmp = y_pre.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,data_iter):
    # 计算准确率
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(sumAccuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

class Accumulator:
    def __init__(self,n):
        self.data = [0.0]*n

    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]

    def reset(self):
        self.data = [0.0]*len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
  
