import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

class batchnorm1d(nn.Module):
    def __init__(self,dim) -> None:
        super(batchnorm1d,self).__init__()
        self.t=1
        self.register_buffer('mu',torch.zeros(dim))
        self.register_buffer('var',torch.zeros(dim))
        self.register_buffer('epsilon',torch.tensor([1e-5]))
        self.on=True
        #dim为卷积通道(dim,1,1)为运用broadcast机制，每个卷积通道用一个BN
        self.gamma=nn.Parameter(torch.ones(dim))
        self.beta=nn.Parameter(torch.zeros(dim))
        
    def forward(self,x):
        #训练阶段为on更新方差与均值，测试阶段off不更新
        if self.on:
            with torch.no_grad():
                self.mu+=(torch.mean(x,dim=0)-self.mu)/self.t
                self.var+=(torch.var(x,dim=0)-self.var)/self.t
                self.t+=1
        x_hat=(x-self.mu)/torch.sqrt(self.var+self.epsilon)
        y=x_hat*self.gamma+self.beta
        return y
    
    def turnoff(self):
        self.on=False

    def turnon(self):
        self.on=True


class batchnorm2d(nn.Module):
    def __init__(self,dim) -> None:
        super(batchnorm2d,self).__init__()
        self.t=1
        self.on=True
        self.register_buffer('mu',torch.zeros(dim))
        self.register_buffer('var',torch.zeros(dim))
        self.register_buffer('epsilon',torch.tensor([1e-5]))
        #dim为卷积通道(dim,1,1)为运用broadcast机制，每个卷积通道用一个BN
        self.gamma=nn.Parameter(torch.ones((dim,1,1)))
        self.beta=nn.Parameter(torch.zeros((dim,1,1)))
        
    def forward(self,x):
        if self.on:
            with torch.no_grad():
                self.mu+=(torch.mean(x,dim=[0,2,3])-self.mu)/self.t
                self.var+=(torch.var(x,dim=[0,2,3])-self.var)/self.t
                self.t+=1
        mu_extend=torch.unsqueeze(torch.unsqueeze(self.mu,-1),-1)
        var_extend=torch.unsqueeze(torch.unsqueeze(self.var,-1),-1)
        x_hat=(x-mu_extend)/torch.sqrt(var_extend+self.epsilon)
        y=x_hat*self.gamma+self.beta
        return y
    
    def turnoff(self):
        self.on=False

    def turnon(self):
        self.on=True


class dropout(nn.Module):
    '''
    on决定dropout是否生效，在测试时选择off
    '''
    def __init__(self,p) -> None:
        super(dropout,self).__init__()
        self.register_buffer('p',torch.tensor(p))
        self.on=True

    def forward(self,x):
        if not(self.on):
            return x
        else:
            x=(torch.rand(x.size(),device=x.device)>self.p)/(1-self.p)*x
            return x

    def turnoff(self):
        self.on=False

    def turnon(self):
        self.on=True

# print("----begin test drotput----")
# a=torch.rand((3,4,5))
# drop=dropout(0.5)
# print(drop(a))
# print("----end test drotput----\n\n")

# print("----begin test bn2d----")
# bn=batchnorm2d(3)
# a=torch.arange(0,2*2*3*2*2,2,dtype=torch.float32)
# a=torch.reshape(a,(2,3,2,2))
# bn(a)
# print(bn.var,bn.mu)
# a=torch.arange(0,1*2*3*2*2,1,dtype=torch.float32)
# a=torch.reshape(a,(2,3,2,2))
# bn(a)
# print(bn.var,bn.mu)
# print("----end test bn2d----\n\n")

class NET1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = dropout(0.5)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, dilation=2)
        self.bn2d1 = batchnorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2d2 = batchnorm2d(16)

        self.fc1 = nn.Linear(16 * 16 * 16, 256)
        self.bn1d1 = batchnorm1d(256)

        self.fc2 = nn.Linear(256, 64)
        self.bn1d2 = batchnorm1d(64)

        self.fc3 = nn.Linear(64, 36)
        self.sigmoid= nn.Sigmoid()
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn2d1(x)
        x = self.drop(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2d2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.drop(x)

        x = F.relu(self.fc1(x))
        x = self.bn1d1(x)
        x = self.drop(x)

        x = F.relu(self.fc2(x))
        x = self.bn1d2(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def turnoff(self):
        self.dropoff()
        self.bnoff()
        
    def turnon(self):
        self.dropon()
        self.bnon()
        
    def dropon(self):
        self.drop.turnon()
        
    def dropoff(self):
        self.drop.turnoff()
        
    def bnon(self):
        self.bn2d1.turnon()
        self.bn2d2.turnon()
        self.bn1d1.turnon()
        self.bn1d2.turnon()
        
    def bnoff(self):
        self.bn2d1.turnoff()
        self.bn2d2.turnoff()
        self.bn1d1.turnoff()
        self.bn1d2.turnoff()


class NET2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = dropout(0.5)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, dilation=2)
        self.bn2d1 = batchnorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2d2 = batchnorm2d(16)

        self.fc1 = nn.Linear(16 * 16 * 16, 256)
        self.bn1d1 = batchnorm1d(256)

        self.fc2 = nn.Linear(256, 64)
        self.bn1d2 = batchnorm1d(64)

        self.fc3 = nn.Linear(64, 36)
        self.sigmoid= nn.Sigmoid()
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


ratio=0.7
batch_size=128
transform_pic=transforms.Compose([transforms.Grayscale(),transforms.Resize([80, 80]),transforms.ToTensor()])
trainset=torchvision.datasets.ImageFolder('../archive/Database of 36 handwritten Kyrgyz letters - Train and Test/train/',transform_pic)
testset=torchvision.datasets.ImageFolder('../archive/Database of 36 handwritten Kyrgyz letters - Train and Test/test/',transform_pic)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print('cuda available:',torch.cuda.is_available())
print('cuda number',torch.cuda.device_count())

'''
net=NET2()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))
test_acc_top1=[]
test_acc_top5=[]
train_acc_top1=[]
train_acc_top5=[]
train_loss=[]
test_loss=[]
for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs, labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        with torch.no_grad():
            total += labels.size(0)
            _, predicted = outputs.data.topk(5, 1, True, True)
            labels_resize = labels.view(-1,1)
            correct_top1 += torch.eq(predicted[:,0].view(-1,1), labels_resize).sum().float().item()
            correct_top5 += torch.eq(predicted, labels_resize).sum().float().item()
            
    train_loss.append(running_loss)
    train_acc_top1.append(correct_top1/total)
    train_acc_top5.append(correct_top5/total)

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images,labels=data[0].to(device),data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # choose top-k
            total += labels.size(0)
            labels_resize = labels.view(-1,1)
            _, predicted = outputs.data.topk(5, 1, True, True)
            correct_top1 += torch.eq(predicted[:,0].view(-1,1), labels_resize).sum().float().item()
            correct_top5 += torch.eq(predicted, labels_resize).sum().float().item()
    test_loss.append(running_loss)
    test_acc_top1.append(correct_top1/total)
    test_acc_top5.append(correct_top5/total)
    print("[{}/500,train loss: {},top1: {},top5: {}]".format(epoch+1,train_loss[-1],correct_top1/total,correct_top5/total))
    
torch.save(net.state_dict(), "none_model.pth")
print('Finished Training')
filename='none.pkl'
with open(filename,'wb') as f:
    pickle.dump([train_loss,test_loss,train_acc_top1,train_acc_top5,test_acc_top1,test_acc_top5],f)
plt.figure()
plt.plot(train_acc_top1)
plt.plot(train_acc_top5)
plt.plot(test_acc_top1)
plt.plot(test_acc_top5)
plt.legend(['train_acc_top1','train_acc_top5','test_acc_top1','test_acc_top5'])
plt.savefig('none.png')
plt.show()
'''

net=NET1()
# if torch.cuda.device_count() > 1:
#     net=torch.nn.DataParallel(net,device_ids=[0,1,3,4,5,6,7])
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))
test_acc_top1=[]
test_acc_top5=[]
train_acc_top1=[]
train_acc_top5=[]
train_loss=[]
test_loss=[]
for epoch in range(500):  # loop over the dataset multiple times
    net.turnon()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs, labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        with torch.no_grad():
            total += labels.size(0)
            _, predicted = outputs.data.topk(5, 1, True, True)
            labels_resize = labels.view(-1,1)
            correct_top1 += torch.eq(predicted[:,0].view(-1,1), labels_resize).sum().float().item()
            correct_top5 += torch.eq(predicted, labels_resize).sum().float().item()
            
    train_loss.append(running_loss)
    train_acc_top1.append(correct_top1/total)
    train_acc_top5.append(correct_top5/total)

    net.turnoff()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images,labels=data[0].to(device),data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # choose top-k
            total += labels.size(0)
            labels_resize = labels.view(-1,1)
            _, predicted = outputs.data.topk(5, 1, True, True)
            correct_top1 += torch.eq(predicted[:,0].view(-1,1), labels_resize).sum().float().item()
            correct_top5 += torch.eq(predicted, labels_resize).sum().float().item()
             
    test_loss.append(running_loss)
    test_acc_top1.append(correct_top1/total)
    test_acc_top5.append(correct_top5/total)
    print("[{}/500,train loss: {},top1: {},top5: {}]".format(epoch+1,train_loss[-1],correct_top1/total,correct_top5/total))

torch.save(net.state_dict(), "drop_model.pth")
print('Finished Training')
filename='drop.pkl'
with open(filename,'wb') as f:
    pickle.dump([train_loss,test_loss,train_acc_top1,train_acc_top5,test_acc_top1,test_acc_top5],f)
plt.figure()
plt.plot(train_acc_top1)
plt.plot(train_acc_top5)
plt.plot(test_acc_top1)
plt.plot(test_acc_top5)
plt.legend(['train_acc_top1','train_acc_top5','test_acc_top1','test_acc_top5'])
plt.savefig('drop.png')
plt.show()