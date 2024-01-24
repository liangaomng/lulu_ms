import torch
import torch.nn as nn
import os
import numpy as np
from abc import  abstractmethod
from torch.utils.data import TensorDataset, DataLoader
class Produce_Data_set(): #训练集，验证集，测试集 7:2:1
    def __init__(self, x_min=-1,x_max=1,num=5000, mu=1,seed=0):
        self._num= num
        self._mu = mu
        self._x_min = x_min
        self._x_max = x_max
        np.random.seed(seed)

    def target_func(self,x):
        out = np.sum(np.exp(-x ** 2) * np.sin(self._mu * x ** 2), axis=-1, keepdims=True)
        return out
    
    def produce_data(self,path):
        #文件夹
        if not os.path.exists(path+f"mu_{self._mu}"):
            os.makedirs(path+f"mu_{self._mu}")

        x = np.random.uniform(self._x_min, self._x_max, self._num)[:, np.newaxis].astype(np.float32)
        y = self.target_func(x)

        # 将 NumPy 数组转换为 PyTorch 张量
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)

        # 划分数据集：训练集，验证集，测试集为 7:2:1
        train_size = int(0.7 * self._num)
        val_size = int(0.2 * self._num)

        train_data = TensorDataset(x_tensor[:train_size], y_tensor[:train_size])
        val_data = TensorDataset(x_tensor[train_size:train_size + val_size], y_tensor[train_size:train_size + val_size])
        test_data = TensorDataset(x_tensor[train_size + val_size:], y_tensor[train_size + val_size:])


        # 创建 DataLoader

        #保存
        torch.save(train_data, path+f"mu_{self._mu}"+'/train_loader.pt')
        torch.save(val_data, path+f"mu_{self._mu}"+'/val_loader.pt')
        torch.save(test_data, path+f"mu_{self._mu}"+'/test_loader.pt')

import matplotlib.pyplot as plt

produce_Data_set= Produce_Data_set(mu=70)
produce_Data_set.produce_data(path="../data/")
path_to_data="../data/mu_1/val_loader.pt"
# 加载数据集
train_data = torch.load(path_to_data)

# 将TensorDataset转换为numpy数组
x_train = train_data.tensors[0].numpy()
y_train = train_data.tensors[1].numpy()

# 绘制数据点
plt.scatter(x_train, y_train, s=1)  # s是点的大小
plt.title('Training Set Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

