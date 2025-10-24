w1 = 0.15
w2=0.2
w3=0.25
w4=0.3
w5=0.4
w6=0.45
w7=0.5
w8=0.55
b1=0.35
b2=0.6
import torch
from torch.nn import Sigmoid
i1=0.05
i2=0.1
h1=w1*i1+w2*i2+b1
h2=w3*i1+w4*i2+b1
outh1 = Sigmoid()(torch.tensor(h1))
outh2 = Sigmoid()(torch.tensor(h2))
o1 = w5*outh1+w6*outh2+b2
o2 = w7*outh1+w8*outh2+b2
print("Output of neuron o1:",o1.item())
print("Output of neuron o2:",o2.item())
