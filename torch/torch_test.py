from __future__ import  print_function
import torch
from torch.autograd import Variable

def fn_add(x,y):
    x=x+y

x=5
y=10
fn_add(x,y)
print(x)