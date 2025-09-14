import torch
import numpy as np
print(complex(real=4))# imag unless specied , 0
# print(int(4+9j)) error
data=[[2,3],[5,4]] # like numpy , this must be homogenous
data2=[[3,5,True],[1,2.4,2+9j]] # string is NOT suppoted
dataT=torch.tensor(data=data)
dataT2=torch.tensor(data=data2)
# print(dataT2)# complex is typecasting everything
print(type(dataT),dataT.shape,dataT2.type())  #,torch.size(dataT) not works,# numpy allows str
arr=np.array([[1,2,3],["hi",True,5+8j]]) # why array.ndarray not wokkring
# dataBrigeTensor=torch.from_numpy(arr) , not work with str
#from_numpy also make linked copy
newT=torch.ones_like(dataT) 
newTTTT=torch.ones_like(dataT,dtype=complex) # if dtype specified , converts all the data in that , here shape of dataT is retained but dtype is NOT
newTTT=torch.rand_like(dataT,dtype=float)# int is NOT sensible cuz randlike only generates BETWEEN 0 and 1 , so if force the rand to override dtype for INT only  then NO range of function
newTT=torch.randint_like(newT,low=0,high=9) # both inclusive
print(newT,newTT,newTTTT,newTTT,sep="\n")
rand_tensor = torch.rand(size=(shape:=(2,3)))
one_tensor=3*torch.ones(shape) # ones or torch.zeros
print(rand_tensor,one_tensor,rand_tensor.shape)
#diff bw .ones and .ones_like boils down to shape inheritance versus explicit specification
print((2,3)==(2,3,))
print((2)!=(2,)) # (2,) is TUPLE and (2) is INT
print(torch.cuda.is_available())
print(dataT2.device) # where the tensor is stored
# slicing is SAME as numpy , also the special slicing [a][b][c]=[a,b,c] too # not for list
print(torch.cat((torch.tensor(((2,3),(1,9))),torch.tensor([[0],[0]])),dim=1)) #carefully note the tesnor2 in each case
print(torch.cat((torch.tensor(((2,3),(1,9))),torch.tensor([[0,0]])))) # dim 0 is default 
# torch.cat([tensors=iterable of tensors]) dim 0 is row , dim 1 is coulmn and so on
#checkout torch.stack # .tensor .Tensor
a=torch.tensor(x:=[[0,2],(3,4),[1,1]]).mul(torch.tensor([[1,2],[3,1],[9,8]]))# element wise element or using *
print(a,a.T,sep="\n") # note properly the trnaspose
print(a.matmul(a.T)) # matmul for MATRIX multiplication #.T for transpose
#tensor.operator is change temp and return and tensor.operator_ is in place i.e change the original # saves memory but not encouraged for derivatives cuz loss of history
print(a.matmul(a.T)==a@a.T) #lmfaooo , so yeah @ is also for matrix multiplication
one=torch.ones([2,2])
one_arr=one.numpy() # tensor to array
one.add(4)
print(one,one_arr)
one.add_(4)
print(one,one_arr) # always linked