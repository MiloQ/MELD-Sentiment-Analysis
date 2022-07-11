import torch.nn as nn
import torch

"""
无关痛痒的测试一些语法的文件

"""
x = torch.tensor(torch.rand(4,512)) # sqLen* batchsize * embedding
head_shape = x.shape[:-1]
print(head_shape)
print(*head_shape)
for (i,_) in range(50):
    print("sjakl")