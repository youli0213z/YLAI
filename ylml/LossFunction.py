import torch

def MSE(x, y, device='cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, device=device)
    if torch.is_tensor(y) == False:
        y = torch.tensor(y, device=device)


    return torch.matmul((x - y).T, (x - y)) / x.shape[0]
def BCE(x,y,device='cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, device=device)
    if torch.is_tensor(y) == False:
        y = torch.tensor(y, device=device)
    l = 0
    for i in range(x.shape[0]):
        l -= (torch.log(x[i])*y[i] + torch.log(1-x[i])*(1-y[i]))
    return l

def CrossEntropy(x, y, class_num, device='cpu'):
    if torch.is_tensor(x) == False:
        x = torch.tensor(x, device=device)
    if torch.is_tensor(y) == False:
        y = torch.tensor(y, device=device)
    l = 0
    for i in range(x.shape[0]):
        l -= torch.log(x[i][y[i]])
    return l
