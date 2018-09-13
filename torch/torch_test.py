from __future__ import  print_function
import torch
from torch.autograd import Variable

x=Variable(torch.ones(2,2),requires_grad=True)
y=x+2
z=y*y*3
out=z.mean()
print(z,out,x,y)