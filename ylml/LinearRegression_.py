from ylml import  ActivationFunction,LossFunction,LinearRegression,SGD,Trainer
from sklearn.datasets import load_boston
import numpy as np
import torch
import pandas as pd

if __name__ == '__main__':
    x = np.random.randint(-100,100,[100,1])
    # print(x.shape)
    y = x[:,0] * 3
    y = torch.tensor(y).unsqueeze(1)
    # print(y.shape)
    # y += np.random.randn([100,1])
    # df = pd.read_csv('kc_train.txt')
    # x = torch.tensor(df.iloc[0:500,3:5].values)
    # x_ = torch.ones([x.shape[0],x.shape[1]+1])
    # x_[:,1:] = x
    # y = torch.tensor(df.iloc[0:500,-1].values,dtype = torch.float32)
    # y = y.unsqueeze(1)
    #x = torch.tensor([[1,2],
                      # [2,4],
                      # [3,6]])
    x_ = torch.ones([x.shape[0],x.shape[1]+1])
    x_[:,1:] = torch.tensor(x)
    # y = torch.tensor([[3],
    #                   [6],
    #                   [9]])
    # y = y.unsqueeze(1)
    max_epoch = 100
    batch_size = 50
    linearRegression = LinearRegression(100,1)
    optimizer = SGD()
    trainer = Trainer(linearRegression, optimizer)
    trainer.fit(x_, y, max_epoch, batch_size, eval_interval=10)
    trainer.plot()



    # x = ActivationFunction.Softmax([[-1,2,3,4],[5,-1,7,8]])
    # y = [2,2]
    # print(LossFunction.CrossEntropy(x,y,4))