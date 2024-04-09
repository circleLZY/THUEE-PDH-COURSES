import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HW2_sampled import *

df=pd.read_csv('heart_2020_cleaned.csv')
_label=df[df.columns[0]]
label = np.zeros(len(_label))
for i in range(len(_label)):
    if _label[i] == 'Yes':
        label[i] = 1
label = pd.DataFrame(label)
label = label[label.columns[0]]
data=df.drop(columns=df.columns[0])
divide=0.95    
p_size=5000
n_size=5000
test_data=data.iloc[int(divide*len(data)):]
test_label=label.iloc[int(divide*len(data)):]

record=np.load('record.npy')
merit=['acc','precision','recall','oob']
#绘制热力图
for i in range(4):
    plt.figure()
    plt.imshow(record[:,:,i])
    plt.tight_layout()
    plt.xlabel('feature num')
    plt.ylabel('tree num')
    plt.xticks(np.arange(6),labels=np.arange(3,9,1))
    plt.yticks(np.arange(7),labels=np.arange(1,14,2))
    plt.title(merit[i])
    plt.savefig(merit[i]+'.png')
#各指标随树的数目的变化
plt.figure()
x=range(1,14,2)
for i in range(4):   
    plt.plot(x,record[:,0,i])
plt.legend(merit)
plt.xlabel('fig1:num of tree')
plt.savefig('tree.png')
plt.show()
#各指标随特征数目的变化
plt.figure()
x=range(3,9,1)
for i in range(4): 
    plt.plot(x,record[4,:,i])
plt.legend(merit)
plt.xlabel('fig2:num of feature')
plt.savefig('feat.png')
plt.show()

#绘制AUPRC曲线
iternum=5
nmin=30
featnum=7
treenum=3
recall=[]
prec=[]
eps=1e-6
for m in range(iternum):
    train_data=data.iloc[:int(divide*len(data))]
    train_label=label.iloc[:int(divide*len(data))]
    train_data_p=train_data[train_label==1]
    train_label_p=train_label[train_label==1]
    train_data_n=train_data[train_label==0]
    train_label_n=train_label[train_label==0]
    p_index=random.sample(range(len(train_data_p)),p_size)
    train_data_p=train_data_p.iloc[p_index].reset_index(drop=True)
    train_label_p=train_label_p.iloc[p_index].reset_index(drop=True)
    n_index=random.sample(range(len(train_data_n)),n_size)
    train_data_n=train_data_n.iloc[n_index].reset_index(drop=True)
    train_label_n=train_label_n.iloc[n_index].reset_index(drop=True)
    train_data=pd.concat([train_data_p,train_data_n])
    train_label=pd.concat([train_label_p,train_label_n])

    cla=RF(nmin,featnum,treenum)
    cla.train(train_data,train_label)
    _recall=[]
    _prec=[]
    for th in np.arange(0,1.1,0.1):

        pred=[]
        for n in range(len(test_data)):
            pred.append(cla.predict(test_data.iloc[n],th))
        TP=0
        FP=0
        FN=0
        for idx,real in enumerate(test_label):
            if real==1 and pred[idx]==1:
                TP+=1
            if real==0 and pred[idx]==1:
                FP+=1
            if real==1 and pred[idx]==0:
                FN+=1  
        _recall.append(TP/(TP+FN+eps))
        _prec.append(TP/(TP+FP+eps))
    print('_recall:', _recall)
    print('_prec:',_prec)
    recall.append(_recall)
    prec.append(_prec)

recall=np.mean(recall,axis=0)
prec=np.mean(prec,axis=0)

plt.figure()
plt.plot(recall,prec,'.-')
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('AUPRC.png')
plt.show()