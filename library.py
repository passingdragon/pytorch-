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
