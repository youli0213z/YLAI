import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from ylml.LossFunction import MSE
from ylml.Optim import SGD,MB_SGD
from ylml.ml import Train,LinearRegression_,LinearRegression,LinearRegDataset
import numpy as np


#构建线性回归数据集
# x = torch.randint(-100,100,[500,3],dtype =torch.float32,requires_grad=True )
# print(x.shape)
# y = x[:,0] * 3 + x[:,1] * -1 + x[:,2] * 20
# y = torch.randn(500,requires_grad=True)
# y = y.unsqueeze(1,)
# print(y.shape)
# x_train , x_val = x[0:400,::] , x[400:,::]
# y_train , y_val = y[0:400,::] , y[400:,::]
# class Dataset_(Dataset):
#     def __init__(self,x,y):
#         self.x = x
#         self.y = y
#     def __getitem__(self, idx):
#         return self.x[idx,::],self.y[idx,::]
#     def __len__(self):
#         return self.x.shape[0]
# train_set = Dataset_(x_train,y_train)
# val_set = Dataset_(x_val,y_val)
# train_dataloader = DataLoader(train_set,batch_size = 128,shuffle = True)
# val_dataloader = DataLoader(train_set,batch_size = 128,shuffle = True)
# device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
# w = torch.randn([3,1],requires_grad=False).to(device)
# w.requires_grad = True
# b = torch.zeros(1,requires_grad=False).to(device)
# b.requires_grad = True
# print(w,b)
# model = LinearRegression_(w,b)
# linearRegression = LinearRegression().to(device)
linearRegdataset = LinearRegDataset(500,1,3,1,128)
linearRegdataset.show_dataset()
# loss_function = LossFunction.MSE
# optimizer1 = Optim.SGD(model.parameters(),128,0.0001)
# optimizer2 = Optim.MB_SGD(model.parameters(),0.0001)
# optimizer3 = torch.optim.SGD(model.parameters(),0.0001)
# #train = Train(100,0.0001,loss_function,optimizer1,model,task_type = 'REG',device = device)
# train = Train(10,loss_function,optimizer3,model,task_type = 'REG',device = device)
# #train.start_train1(train_dataloader,3,[w,b],128)
# params = train.start_train(train_dataloader,3)
# #train.start_train2(train_dataloader,3,linearRegression.parameters())
# #train1.start_train(train_dataloader,3)
# train.show_loss_value()
# print(model.parameters())


