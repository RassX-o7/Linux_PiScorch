import torch # NOTE for working of CNN , refer to the VOICENOTE in device
import torch.nn as nn # the working of cnn is discussed seperately
import torch.nn.functional as F
from torchinfo import summary
# consider running C:\Users\Rachit Soni\Downloads\convolution_torchnn.png
# for reference , 

class Net(nn.Module):
    def __init__(self):
        # super(Net,self).__init__()  need to explictly define the parent class , older method
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5) # input channel , output channel , (size of kernel) , hence initalize 6 kernels of 5*5 each
        self.conv2 = nn.Conv2d(6,16,5) # here bias  is of size no. feature maps , one bias scalar for each map , then added to each pixel in map 
        #16 kernels each of size 6*5*5 , note in convolution whole (all channels are passed at same time i.e as a bulk 3d image)
        # an affine operation: y = Wx + b , affine means Linear , the op itself is Linear , the activation function is apllied after
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        c1=F.relu(self.conv1(input))
        s2=F.max_pool2d(c1,(2,2))
        c3=F.relu(self.conv2(s2))
        s4=F.max_pool2d(c3,kernel_size=2) # output 16*5*5 # if square kernel may just use kernel size = n
        s4=torch.flatten(s4,1) 
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output
#NOTE the OUTPUT is supposed to be [1,10] a row vector 
def dummyx(arg):
    pass
def typehint(x) -> int : 
    arg=dummyx(x)
    return x
out=typehint("hi") #hover over out, and do without typehint now
#torch.nn.Conv2d is a class
net=Net()
print(net)
# summary(model=net,input_size=(1,1,32,32),verbose=0) #no print
# summary(model=net,input_size=(1,1,32,32),verbose=1) # some info
summary(net, input_size=(1, 1, 32, 32), verbose=2) # (bacth,n channels ,height width) as dummy input , dummy
params=net.parameters()
print(type(params))
params=list(params)
print(len(params))
print([z.shape for z in params]) # list comprehension inside [] , without it , it is a generator
# print(params) , list

#insider , the only way forward is getting called itself is because teh __call__ inherited from nn.Module works only for name `forward` method other names not work , an object can be evoked by a method but with object.foorward() but if __call__ then object() directy evoked
randm_input=torch.randn(size=[1,1,32,32]) #torch.rand_like(tensor) convert tensor to randlike overide , randn generates randm 
output=net(randm_input)
outputX=net.forward(randm_input)
print(output)
# after each forward pass , we must set the grad equal to 0 so that they can be computed again from scartch for each loss , cuz param.grad is accumicaled 
net.zero_grad() # or optimizer.zero_grad()
output:torch.Tensor # you can do this type hinting to recognize the backward in 54 , but 49 is direct BUT never DO .forward() i.e call forward yourself  instead you should evoke the object and it would execute __call__ (a hook) that runs various system checks and etc before and after the forward pass , also runs the forward pass automatically
output.backward(gradient=torch.randn([1,10]))
# outputX.backward() this is getting recognized means the problem was in the __call__ shiz , btw we want to backpropogate the error not the output

# NOTE torch.nn  optimized to perform calcs parallel so many samples at once so if the data is 3d , n channels , height , width  , do tensor.unsqueeze[0] adds fake dimension of size 1 for faster computation
# cuz single sample not supported , if input data is 4d tensor then unsqueeze at 1 dimension at start 5d 

# when __call__ is exectuted it checks the regsiters for any added hooks(functions) to be exectuted beforehand or afterward , it handles everything , the main function is to call forward
# __call__ benifical helps to modify something temp without messing with the model structure
# for example in model pruning , if have a big pre trained model and want it to make it more mobile for specific lineups etc by removing the unnecessary weight that adds to the compute without SIGNIFICANT loss so we need a forwards "pre" hook that needs to be registered
#You attach a pruning hook to your layer: layer.register_forward_pre_hook(pruning_function) , it sets few weight to 0 , when we call a object/instance like a function , it looks for this __call__ and exectutes whats inside it
# nn.Module.register type this to know about various registers and their tasks 

# defining the error function , target is expected outcome
target=torch.randn(10) # can directly do [1,10] but gaan masti  , insight , it is *size , so can directly type out size dimensions but if i use size = then must use sequence
target=target.view(1,-1) # *args is shape , (tho can explictly put size =  ) and dtpe is 1 kwargs  # here view represent , a read only copy of the tensor as specified shape , [1,-1] -1 is a place holer to automaticaly decide it , since one dimension is 1 and 1*x = 10 then x is 10 , so view returns tensor as [1,10]
# can do new method .reshape() 
targetX=target.reshape(shape=(1,-1)) 
print(targetX==target.reshape(1,10))
criterion=nn.MSELoss() # class 
loss=criterion(outputX,target) # so yes , here too *shape can be used but if shape= then sequence req as *var represent unpacked , var represnt packed i.e sequence
print(loss)
print(loss.grad_fn) # grad_fn looks at this childs history , # since params with req grad = true are the starting ponints and start tracking the childs
loss:torch.Tensor

print(loss.grad_fn.next_functions[0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # keep backwards propogating

T1=torch.tensor([1,2,3],requires_grad=True,dtype=float) # NOTE , float is IMP , otherwise if do INT , then DO NOT WORK , try , require grad is only applicatble to complex and float dtypes
T2=torch.randn(1,3,requires_grad=True)
print(T3:=(T1+(T2**2)))
T4=(T3*T1)*(T2+T1)
print(T4.grad_fn) # since T4 was last created by mul , as a*b 
print(T3.grad_fn) # T3 is x+y
print(T4.grad_fn.next_functions) # T4 last operation was addition , whom was it bw ? (T3*T1) and (T2+T1) so next function now tells us about the operands of the last operation(mul) it is mul again T3*T1 and T2+T1 (ad) note if was single , then still addition(+0) like x= a*(b+c)
print(T4.grad_fn.next_functions[0][0].next_functions) # this is interesting
# T4 is created by mul => mul and add  # in above line , we want to see about the mul operands (T3 and T1) so we do nest_functions[0][0] first 0 was for to select bw mul and add , now in [0] we have two things mul object backward ( bw t3 and t1) and 0
# now if move backward form that mul object too (t3 * t1) get operand t3 and t1 history , if nextfunctions[0][0].nextfunctions > now we get about t3 and t1 history , so what is t3 , t3 is t1 + t2**2 , so its a ADDback object and what about T1 , (t3*t1) its a accumalted object MEANS , a standalone tensor , so backtrace from there

T5=T1+T2
print(T5.grad_fn)
print(T5.grad_fn.next_functions)
T5=T5+2
#there is no option in tensore to append or pop , tesnors are immutable in form of shape i.e when created it creates a continous memory space , for faster calcuations but only values can be changed , so yeah NO append and shit
#but can create new tesnor and ca concatenate to form pusedio append but create new tesnor that way 

print(T5.shape)
temp=torch.Tensor([0.98])
print(temp.shape) # HERE ITS ONE [1] but to concatenate to [1,3] atleast need 2d
temp=torch.Tensor([0.98,]) # NO does not count as [1,1] only way is [[]]
print(temp.shape)
temp=torch.Tensor([[0.98]]) # here dimension is 1,1 since added the brackets 
print(temp.shape)
# dim 1 is imp , row column 0 , 1
T6=torch.cat([T5,temp],dim=1)
print(T6,T6.shape)
t1,t2=torch.chunk(T6,chunks=2,dim=1) #bro why dont you gte it that dim is by default 0 means it will try to split the rows , which is 1 in two parts
# set dim to 1 ,# returns TWO things
print(torch.chunk(T6,2)) # exact same

T=t2**2
print(T.grad_fn) # pow back
print(T.grad_fn.next_functions) # tells about operands of pow means t2
#t2 is made by function chunk BUT , chunk returns  two tensors , so this index ((<SplitBackward0 object at 0x7b08f312bb50>, 1),) that 1 is used to specify WHICH output
#now refer to after 91
  # # # # Actually  for normal operation add sub , the output is one only so output_index is 0
# backprop comes
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
#net is object of class and Net is that class  ,conv1 was the attribute of object , see forward method , self.conv1 is the attrubte whose equal to a object of conv2d class , .bias is attr of that conv object WHICH IS A TESNOR, then .grad is atr of that tensor , any tensor has grad attr

#updating params
lr=0.01
for f in net.parameters(): # net.param is generator , calling it each time gives next value
    f.data.sub_(f.grad.data*lr)

"to understand what happened , consider this"
# t1=torch.tensor([1,2],requires_grad=True) # this will throw error cuz only float and complex can have 
t1=torch.tensor([1,2],requires_grad=True,dtype=float) 
t2=(3*t1) + (t1**2)/10
print(t2.grad_fn)
print(t2.grad_fn.next_functions)
print(t2.grad_fn.next_functions[0][0].next_functions)
print(t2.grad_fn.next_functions[0][0].next_functions[0][0])
print(og:=t2.grad_fn.next_functions[0][0].next_functions[0][0].variable,id(og))
# print(t2.grad_fn.next_functions[0][0]._saved_tensors)
# print(t2.grad_fn._saved_tensors)
# print(t2.grad_fn.next_functions[0][0].variable)
# print(t2.grad_fn.variable)

# forget E V E R Y T H I N G
"start fresh"
#A DAG is created to store pointers and operands type
# Tensor A req grad is true 
A=torch.Tensor([1,9])  # NO Tensor([1,9],requires_grad=True) , no kwarg , thats why .tensor() used , but .Tensor autoset float
A.requires_grad_(True)
G=torch.tensor([0.1,2.1],requires_grad=True) # not truthy values , only bool
B = (A**2)+G
C=A+B
D=(A*2 + (B*G)) - C
E=D/2 + A
"""
what happens , you defined a tensor A , now all calulations on it aare tracked , grad_fn returns the last operand
E,grad_fn returns add , add.next_fn retrurns the graph of operand of this add 
next_fn [0][0] now returns divison object , BUT next_fn[1][0] is an ACCUMULATED OBJECT , when USED variable attribute it gives us the tensor on which the value was used as operand ,simply tensor A """
print(type(sm:=E.grad_fn.next_functions[1][0].variable),sm == A , sm is A) # sm is A is also true cuz , it points to the sam eexact unchanged var
"""
now cehck for next_fn[0][0] its the division object cuz D/2 was used , again check teh operands of Div (there must be cuz iyts NOT accumulated)"""
print(E.grad_fn.next_functions[0][0].next_functions) # operand of D div 2 are D and 2 (None) , now when D is checked we check its history because D means nothing , its a object D is composed of same ol add/sub objects , its a subObject ,D=A*2 + (B*G) - C last one is used NOTE if it was + tehn add object but IF mul then ADD object , maybe priority
"now sub object D , still not a independent tensor  , check operand of this object"
print(E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions) # BTW CAN   D I R E C T L Y   check D.grad_fn ,  but we back trace from END 
"""
D=[A*2 + (B*G)] - [C] backward from sub object yields to operand object a*2 + b*g , add object and back trace from C a + b also add object "
check for operand of C first
"""
print(E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions) 
"""
yields about C = A + B , A is leaf , cant be back traced so its a accumulated object rest B is back traced to  YK 
"""
#NOW comes the real deal 
# reASSignment 

print(E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].variable is A)
print(E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[1][0].variable is G)
# if check graph then actuall correct but not acc positions index , since will get tangled badly , so don tblindly follow graph
from torchviz import make_dot
graph = make_dot(var=E,params={A.data:A,"G":G,f"E : {E.data}":E,"D":D}) # try with A 
graph.render("computation_graph4", format="png") # png or pdf many formats

graph = make_dot(var=E,params={f"A:{A.data}":A,  f"G:{G.data}": G ,f"E : {E.data}":E,"D":D}) # note intermetiate D , E B does not matter only accumulated i.e indepeendednt tensors matter  
graph.render("FINAL", format="jpg")

"""
NOW reassignment, currenly the acculmulated_grad.variable POINTS to same as OG A AND G unchanged
"""
print(A,A+9)
ogA=A.clone()  # a is tensor , b=a c=a.clone()   a[1]= 9 print , both a and b is changed cuz b is referenced to a (not a abosulte vlaue )
A=A+9 # what this DOES , created a NEW tensor A+9 and the pointer to A was now switched to A(name reference changed) 
print((sm:=E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].variable) is A)
print(sm==ogA) # still points to the OG tensor object even tho the var is not associated with it now
print(sm) # its cuz the graph was made with value [1,9] which happened to be "A"at that time , not its not , BUT the next variable still points to [1,9]
print(E.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
X=A+B 
try:
    print(X.grad_fn.next_functions[0][0].variable) 
    "saw what happened , not exec , CUZ A is itself pointing to sm , A itself (THE OG) , when you say A = A+9 the rhs is concluded first so A+9 is first mad ea tesnor , whose object is add then that uhhhhhhhhhhhhhhhhh"
except:
    print(X.grad_fn.next_functions[0][0].next_functions[0][0].variable)

# the (2) ig repesent the shape/size of tensor
print(A)
graph = make_dot(var=X,params={f"A:{A.data}":A,  f"G:{G.data}": G ,f"E : {E.data}":E,"D":D,"X":X}) # note intermetiate D , E B does not matter only accumulated i.e indepeendednt tensors matter  
graph.render("FINALxx", format="jpg")

# YOU can KINDA say it got confues , check A node cant decide

for f in net.parameters(): 
    f.data.sub_(f.grad.data*lr)
#.data
# one thing ,  += and = + ARE NOT same
# x+=1 is INPLACE ,x = x+1 is REASSIGNMENT

#reassignment and inplace both dangerous without .data or no_grad()
#inplace instantly breaks the graph cuz changes the root tesnor
# one before anything is that DAG keeps track of the tensor object NOT the var name
#T=tensor([1,2,3],require_gradn = true
#creates tensor object t1 AND the T var name points to it , DAG build up , if reassignment done 
#T= T+2 , T+2 is new TENSOR object t2 , and now the var name T points to t2 NOT t1 , but t1 is STILL stored in memory  cuz its been tracked by autograd , meaning , THE DAG is keeping track of tensor objects not names , T= T+2 is stored as new add back object and if doo .variable , it will show same t1 , refer to below example

ta=torch.tensor([1.,2.],requires_grad=True)
tb=ta+3
ta=ta*3
tc=ta/5
print(ta.grad_fn) # ta is the NEW tesnor created from the OG tensor **OBJECT** whats why grad_fn is none
print(ta.grad_fn.next_functions[0][0].variable) # this is ta = ta+3 , the operand here ta is that og tensor object 
print(tc.grad_fn.next_functions[0][0]) # no variable attr , cuz that tensor object tc = m /5 , m is itself backtaced to mul back SO the graph is consistenet

#inplace is automatically dangerous cuz it corrups the graph enitirely T +=2 or T.add_(2) changes in the root memory , no new refernece , it changes the tensor object itself so DAG confues which object to refer to since the og is changed and the prev graph was made using the of tensor object oif now point to changed tensor then calculation wont be true

"""Re-assignment does not break the computation graph, but it does break the connection to the optimizer.

## The Problem: The Optimizer Loses Track
Here is the sequence of events:

Setup: When you create your optimizer, you give it a list of parameters to watch and update.
optimizer = optim.SGD(net.parameters(), lr=0.01)
The optimizer now holds a reference to the original tensor objects for all the weights and biases (e.g., it's tracking Tensor_A).

Re-assignment: You perform a manual update in your training loop.
param = param - lr * param.grad
As we established, this creates a new tensor object (Tensor_B) and makes the name param point to it.

The Disconnect: Your model's layer (self.conv1.weight) now refers to Tensor_B (SINCE the param IS the variable and the optimizer when teh params where initalized pointed to THE og TESNOR OBJECT in which req_grad = True). However, the optimizer is still holding a reference to the old, original Tensor_A. It does not know that you've made a new tensor.


"""
####################################
"""
Refresher : torch.autograd  is sepearte package linked to torch.Tensor class for req_grad = true , autograd is what tracks each upadte to build the graph , but theres a problem , autograd needs an unmodified value of that leaf param that "gets changed" through operations , not changed but new tesnors created by changes/operations  
autograd constructs the graph of changes... but if an inplace subtract is used without .data attr , the DAG is fucking correupted , cuz it changes the param at its root memory , this fucks up the graph and autograd gets confused 
Bro .data helps us get the raw data , without autograd tracking it , bypass autograd for that step


1.Problem with f = f - f.grad * lr
This line creates a new computation. When you re-assign f = ..., you are replacing the original parameter tensor (which the optimizer is tracking) with a completely new tensor. The optimizer will lose its reference to the original parameter and the model will not learn (refer to last line of code , > NO) AS WE SAW , the model got confused to the og refernce in graph

2.Problem with f.sub_(f.grad * lr)
This is even more dangerous. f is a "leaf" tensor with requires_grad=True. Performing an in-place operation (.sub_()) on a leaf tensor that requires gradients is forbidden because it can corrupt the information autograd needs for the backward pass. PyTorch will stop you with a RuntimeError.

"""
# so final regards
# we cant use param -= param.grad_fn * lr OR reassignment or similar sort OR inplace _   >> 2 inplace method , 1 manual re assignment 

# only two correct ways 
"Method 1"
for f in net.parameters(): 
    f.data.sub_(f.grad.data*lr) #Or use other inplace method
#CUZ thegraph is built , backward is called , now comes the change , we bypass autograd by chanhing the data of tesnor , autograd does not know , so now the variable must be updated / means the object itself must be changed SO that the new graph in next iterations is built upon the NEW object 
# so NO reassignment after using .data / no grad()

"Method 2"
for f in net.parameters(): 
    with torch.no_grad():
        f.sub_(f.grad*lr) # or other inplace  , no .data req in this
# no_grad pauses the system entirely , but.data bypasses it
# .data is older and unsafe 
#checkout more



#IF WORKING WITH OPTIMIZER
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
temp_input=torch.ones([1,1,32,32])
output = net(temp_input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update auto


"""
The make_dot function takes a params dictionary to label the blue leaf nodes in the graph.
make_dot(final_tensor, params={'A': A}) [ === make_dot(final_tensor)]

When you re-assign A = A + 9:

The computation graph correctly holds a reference to the original A object.

However, the variable name A in your script now points to the new tensor (the result of the addition).

When torchviz tries to render the graph, it looks at the leaf nodes and tries to find their names in the params dictionary. It sees the original A object in the graph but can't find it in your params dictionary (because the name A now points to a different object). This mismatch causes the rendering to fail.
# so it essentailly behaves when you DO NOT give any param dict it just renders out the graph without ANY label
"""
# random question
#When we create a tensor object( t = torch.tensor()) so the underlying tensor(value) is accessed by .data or the object itself is the tensor value , because I have checked when we print the tensor object it prints the value as well as the grad fn and many things but when we print tensor.data it only prints the tensor as a list 
# also I have questioned what is the difference between torch.tensor and torch.Tensor

"ans1 tesnor is the main object wrapper the data is only accessed by .data " # >#"ans1 tesnor is the main object wrapper the data is only accessed by .data " # > # But the main question remains is when I do for example T+3 ,T * 5 then how does it know I am talking about the data of their tensor because it is a object right
#C H E C K O U T 
#C:\Users\Rachit Soni\Desktop\Python\myarc\gemini_ans.txt
#do more research on tensor() and Tensor()
def forwardX(self, input):
    # Convolution layer C1: 1 input image channel, 6 output channels,
    # 5x5 square convolution, it uses RELU activation function, and
    # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
    c1 = F.relu(self.conv1(input))
    # Subsampling layer S2: 2x2 grid, purely functional,
    # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
    s2 = F.max_pool2d(c1, (2, 2))
    # Convolution layer C3: 6 input channels, 16 output channels,
    # 5x5 square convolution, it uses RELU activation function, and
    # outputs a (N, 16, 10, 10) Tensor
    c3 = F.relu(self.conv2(s2))
    # Subsampling layer S4: 2x2 grid, purely functional,
    # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
    s4 = F.max_pool2d(c3, 2)
    # Flatten operation: purely functional, outputs a (N, 400) Tensor
    s4 = torch.flatten(s4, 1)
    # Fully connected layer F5: (N, 400) Tensor input,
    # and outputs a (N, 120) Tensor, it uses RELU activation function
    f5 = F.relu(self.fc1(s4))
    # Fully connected layer F6: (N, 120) Tensor input,
    # and outputs a (N, 84) Tensor, it uses RELU activation function
    f6 = F.relu(self.fc2(f5))
    # Gaussian layer OUTPUT: (N, 84) Tensor input, and
    # outputs a (N, 10) Tensor
    output = self.fc3(f6)
    return output