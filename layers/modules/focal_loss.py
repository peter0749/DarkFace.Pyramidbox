import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def focal_loss(x, y, n_class, alpha=0.25, gamma=2.0):
    cuda = x.is_cuda

    t = torch.eye(n_class)[y] # to one-hot encoding
    t = Variable(t, requires_grad=False)
    if cuda:
        t = t.cuda()
    p = x.sigmoid()                # from p.3 last section
    pt = (p*t + (1-p)*(1-t)).clamp(1e-8,1-1e-8)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt)**gamma
    loss = -w*pt.log()
    return loss.sum()

if __name__=='__main__':
    x = Variable(torch.randn(3, 5), requires_grad=True)
    y = Variable(torch.LongTensor(3).random_(5), requires_grad=False)
    print(focal_loss(x,y,5))
