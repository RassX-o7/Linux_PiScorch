import torch
import numpy as np
from torchvision.models import resnet18,ResNet18_Weights
model=resnet18(weights=ResNet18_Weights.DEFAULT)
data=torch.rand(1,3,64,64)# size is *arg and generator and other are kwargs , so yeah , these only shapes
labels=torch.rand(1,1000) # for 2d tensor it is batch,classes  , so if 1 image then 1x1000 shape
# prediction=model(data) # here batch is of size 1 so 1 image 3 channels of 64*64 pixels
#forward pass
#The standard format for image tensors in PyTorch is (Batch, Channels, Height, Width), or BCHW. channel first
#Tf or keras use (BHWC) # channel last
# loss=(prediction-labels).sum() # not a actual loss function like MSE, easy , here sum of errors with sign , dummy function
# loss_tensor:torch.Tensor=(prediction-labels) , python static analysis is not always correct , porblem is dynamic language , so use type hinting , so python recognizes it , also loss:torch.Tensor=loss-prediction.sum() wont work , Tensor NOT tesnor , as class
# loss=loss_tensor.sum()
# loss:torch.Tensor
# loss.backward() # backward pass the error
optimizer=torch.optim.SGD(params=model.parameters(),lr=1e-2,momentum=0.9)
# optimizer.step()
# ONLY SINGLE GD STEP , you need a loop to train
# print(prediction)
t1=torch.tensor([1.,9],requires_grad=True) # reuqire grad , it tells autograd to track operations on the tensor
# requires_grad means track ALL operations like +/-* and grad , so its not used during in turing testing / inference , so 
# with torch.no_grad():prediction=model(test_data)
# if req_grad not initalize , do it by in place operator 
# req_grad is False , then it is treated/flaged as non learnable paramter 
# torch.Tensor.requires_grad_(bool) # to change tensor attribute 
# when a tensor is initalized with requires_grad = True , all the operations on its childs are tracked , so that all these when operations led backwards leads to the tensor itself (parent)
t1x=torch.tensor([8.9,5],requires_grad=True)
print("check the history of t1 , techinally there shooulnt be cuz its not the child of anyone , ")
print(t1.grad_fn)# .grad_fn is attribute 
None
t22=t1/9 #
t2=(3*t1)+2 # always store last operator , "addition" and the link "from" adition to backwards is 3*t1 accessed by grad_fn.next_functions[0][0] > mulBackwards , tho rarely used , internal
err=t1**3+t1x*t1
t3=t1/t2
m=2*t1+9
n=9+t1*2
k=(m+n**m)**2 # PowBack
print("test--",m.grad_fn,n.grad_fn,k.grad_fn)
print(f"childs of t1 = {t2.grad_fn},{t22.grad_fn},{t3.grad_fn}  ")
print(f"child of err is {err.grad_fn}")
# The object.grad_fn attribute holds an object which is an instance of a specific class. like MulBackward0 AddBackward0 , etc

a_param=torch.tensor([2.,1.],requires_grad=True)
b_param=torch.tensor([6.,2.],requires_grad=True)
Q_vector=3*a_param**3-b_param**2 # element wise
print(Q_vector)
# Q_vector.backward() # tells autograd to consider a tensor as a child and walk backwards through its family history, and automatically store the gradients of paramters which requires_grad=True in their object.grad attribute
# BUT we have a problem , bro we have encountered scalar loss functions , but where definitely we CAN convert Q into a single quanitiy by a set of rules we decide
#for example a neural network can predict two things , temp of measurmenet and its glow , so say we get two loss functions Q1 AND A2 these errors , we may convert them too L=3*Q1 -log(Q2) to get a final scalar loss, so we get a jacobian then 
#here the gradients will be respect params for Q1 and Q2 both , but say we want dL/dQ , so dl/dq1= 3 and dl/dq2=-1/q2 , say we have currenly a value of q2 for this forward pass = 0.5
# now here gradient argument is tensor([3,-2])
# when we pass this , all the object.grad will also be of 2*1 size , say param A then A.grad = [dq1/da , dq2/da]
Scalar_Loss=Q_vector[0]*-1 + 3*Q_vector[1] + 2
# note since a and b are paramteres 1*2 , means takes input of final two extra dimensions too , one for Q[0] and other for Q[1] thereby , Q[0] is affected by a[0] and b[0] only as Q vectoe as total is written as 3a^3 -b^2 so this is element wise 
print(f"Q_vector is -12,-1 for current now LOSS is {Scalar_Loss}")# we know dL/dQ = [dl/dq1 , dl/dq2] = dL/dQ = [-1 , 3]
loss_grad=torch.tensor([-1,3])
Q_vector.backward(gradient=loss_grad)
#dq/da= 9*a^2 = [36,9]
#dq/db= -2*b = [-12,-4]
#dl/da=dl/dq(loss_grad)*dq/da similarily for b
#[-1,3]*dq/da or dq/db (BTW element wise) , get 
delL_delA=torch.tensor([-36,27])
delL_delB=torch.tensor([12,-12])
# thing to note , we did NOT define the relation bw Q vector and scalr loss , only provided gradients but got the same answer when explictly defined
print(a_param.grad==delL_delA)
print(b_param.grad==delL_delB)
a_param_copy=torch.tensor([2.,1.],requires_grad=True)
b_param_copy=torch.tensor([6.,2.],requires_grad=True)
Q_vector_copy=3*a_param_copy**3-b_param_copy**2
Scalar_Loss_copy=Q_vector_copy[0]*-1 + 3*Q_vector_copy[1] + 2
print(f"loss is {Scalar_Loss_copy}") # here NOTE , if just print Scalar_loss_copy , get 11 + tensor info , idk , why when f string , prints magnitude only , tye print(Scalar_Loss_copy)
Scalar_Loss_copy.backward()
print(a_param_copy.grad==a_param.grad)
print(b_param_copy.grad==b_param.grad)
# now we assumed one to one mapping cuz Q= 3a^3-b^2 , so it was couln wise 
# what if MORE generalized , say Q=[Q1,Q2,Q3...] and parameters X=[x1,x2,x3,..] size of Q and X not neccessarily same, Q1= 3x_2^2 + sin(x_7)+2^x1 some shit , here NOT one to one mapping 
#so when calculate dQ/dX it is NOT the same size as that Q/X cuz if one to one mapping then size of Q and X mjst be same otherwise size is sizeof Q * size of X
#the matrix is of form
#row i is dq_all/dxi J = [[dq1/dx1,dq2/dx1,dq3/dx1],[wrt dx2]]
# NOW consider a externalgradient/loss gradient that is same size as that Q , L = f(Q) L = Q1 +Q2*tan(Q6)+ shit
# now convert make dl/dq vector (same size as that of Q)
#now may call L.backwards() OTHERWISE call Q.backwards(gradient=external_grad) , yields the SAME output , dl/dA is of same size as that of A , since Loss is scalar , and A is only one param then 1 size , in previous param A nad B were two so both size 2 , cuz a[0] and a[1] affects differently , now , we do dl/dx here x is the tensor of parameters
# X = 1*7 , Q = 1*15 total 15 outputs/feature that can nn recognize/predict/classify J matrix = 7*15 L= f(Q) size of 1
# dL/dx1 is dL/dQ(1x15) * dQ/dx1(one row of J matrix > 1*15> first row) so take thetranspose of J matrix and do the DOT product cuz x1 influences many Q i then each Q I influences L and at last each amplification added to output net change in L wrt x1
# the gradients or Vector-Jacobian Product Matrix is J.T @ V , where V is the External Gradient this final gradient matrix is the sam size as that of paramters matrix/list/tensor
torch.autograd #is what computes THIS vector-Jacobian Product
# like when we call backwards , it goes backward to childs famil history and applies chain rule on each consequent step untill it reaches a leaf tesnor , where grad_fn is NONE
# this is in a graph form called directed acyclic graph (DAG)
#^C then C:\Users\Rachit Soni\Downloads\dag_autograd.png blue nodes are leaf tensors , represent diff paramters

#NOTE DAGs are dynamic in PyTorch An important thing to note is that the graph is recreated from scratch, say tracked tensors a and b , so each time u call foward pass each step is tracked on childs of a and b and the graph is created each time with scratch then when backwards called that graph used then discarded immediately and recreated again in next forward pass 
#  after each .backward() call, autograd starts populating a new graph. This is exactly what allows you to use control flow statements in your model; you can change the shape, size and operations at every iteration if needed. say if logic changed for spefifc data if the graph was already static , defined alreday , you cant incorporate that new condition "now" yk

# parameters with requires_grad is False are frozen paramters , It is useful to “freeze” part of your model if you know in advance that you won’t need the gradients of those parameters (this offers some performance benefits by reducing autograd computations).
# when fine tuning we can then freeze most of the models to finetune spefifc paramters for a specif new dataset(limited) , say for 1000 specif images of new elphants ..
from torch import nn, optim
# if param frozen then means  no update/descent calculated
model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters(): # default pretrained so  was might be true idk
    param.requires_grad = False # not using inplace ?

# param.requires_grad = False: This is a direct attribute assignment. It's simple, clear, and very readable, which is why it's commonly used in tutorials.
# param.requires_grad_(False): This is the in-place method. It follows the standard PyTorch convention where methods ending in _ modify the tensor.

#In resnet, the classifier is the last linear layer model.fc , so we replace the layer and train it seperately I THINK
model.fc = nn.Linear(512, 10) # params created by this nn.linear are true / unforzen by default 
# only this layer is unfrozen so when backward call only these gradients used
# Optimize only the classifier
optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9)
# only setup new data is NOT being fed atm

#The same exclusionary functionality is available as a context manager in torch.no_grad() , checkout idk

print(model) # check the feature size of last layer 
#fine tuning is when you take a pre trained model

#nn.Linear(512,1000) is replaced nn.Linear(512,10) 


#Fine Tuning is the process of re-traning a pre trained model on a smaller niche dataset with specific outputs , so its kind of retrainnig the LAST LAYERS only
# imageNet > trained on 1000 classes , earlier layers have increasing layers of complexity fisrt edges , color transitions , curves , randomness , shininess , idk , wrinkles , colors etc 
# last layers ONLY put togther these features and combine it to give the final output , that these features are most of the time same in this class
#so imageNET is since on 1000 DIFFERENT classes (very) it has learn to specifcy high level features , say i wanna classify the type of a elephant , specifc species
# 1. retrain neural network , limited scope to elephants only , need MANY AMNY labeled images of diff elephants  , cost
# 2. a) pre trained model on thousand of everyday things , that was recognized to detect features from ANY image (not limited to those 1000 classes) , re train the final layers to somehow detect specifc features and put them such that out is one of these 7 specicies , WORKS VERY WELL , transfer learning , even tho we do not have MANY labeled images of all specicies but the features can be recognized very quicly
# 2. b) pretrained model on animals only > works EVEN better for new elephants specied
# 3. pre  trained model on diff car , works since features are edges and thinsg ARE common but high level features are not really recognized so somewhat less efficient BUT cost effective