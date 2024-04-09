# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# %%
class TREE:
    def __init__(self) -> None:
        self.label=-1
        self.LeftChild=None
        self.RightChild=None
        #判别标准，变量离散时相等往左树，不等往右树；连续时小于等于往左树
        self.Var=None
        self.VarValue=None
        self.VarDiscreate=None
        self.leave=False
    def print(self):
        print(self.Var,self.VarValue,self.VarDiscreate)
        if self.LeftChild!=None :
            self.LeftChild.print()
        if self.RightChild!=None :
            self.RightChild.print()
    def predict(self,df:pd.DataFrame):
        if self.leave:
            return self.label
        if self.VarDiscreate:
            if df[self.Var]==self.VarValue:
                return self.LeftChild.predict(df)
            else:
                return self.RightChild.predict(df)
        else:
            if df[self.Var]<=self.VarValue:
                return self.LeftChild.predict(df)
            else:
                return self.RightChild.predict(df)

def CART(Data:pd.DataFrame, label:pd.Series, nmin:int):
    root = TREE()
    root.leave=True
    #训练数据规模小于阈值，返回节点
    if len(label) <= nmin:
        root.label = label.value_counts().index[0]
        return root
    #训练数据标注相同，返回节点
    if len(label.value_counts())==1:
        root.label = label.value_counts().index[0]
        return root
    #训练数据某一特征都相同则不可用去除之
    for ColumnName in Data.columns:
        if len(Data[ColumnName].value_counts())<=1:
            Data = Data.drop(columns=ColumnName)
    #训练数据无可用特征，返回节点
    if len(Data.columns)==0:
        root.label = label.value_counts().index[0]
        return root
    #计算各特征基尼系数并选出最大的
    root.leave=False
    VarGini=10
    Var=0
    VarValue=0
    VarDiscreate=True
    for FeatName in Data.columns:
        Feat=Data[FeatName]
        FeatValueSpace=Feat.value_counts().index
        #判断特征离散
        if type(Feat.iloc[0])==str:
            FeatDiscreate=True
        else:
            FeatDiscreate=False
        if FeatDiscreate:
            for FeatValue in FeatValueSpace:
                #样本的特征为FeatValue,分为属于/不属于两类数据
                ValueYes=Feat[Feat==FeatValue].index
                ValueNo=Feat[Feat!=FeatValue].index
                #统计两类数据的标签
                labelYes,labelNo=label[ValueYes],label[ValueNo]
                #统计两类数据的Gini
                Gini1=2*np.mean(labelYes)*(1-np.mean(labelYes))
                Gini2=2*np.mean(labelNo)*(1-np.mean(labelNo))
                GiniFeat=(len(labelYes)*Gini1+len(labelNo)*Gini2)/len(label)
                if GiniFeat<=VarGini:
                    VarGini=GiniFeat
                    Var=FeatName
                    VarValue=FeatValue
                    VarDiscreate=FeatDiscreate
        
        if not FeatDiscreate:
            for FeatValue in FeatValueSpace:
                if FeatValue ==np.max(FeatValueSpace):
                    continue
                #样本的特征为FeatValue,分为属于/不属于两类数据
                ValueYes=Feat[Feat<=FeatValue].index
                ValueNo=Feat[Feat>FeatValue].index
                #统计两类数据的标签
                labelYes,labelNo=label[ValueYes],label[ValueNo]
                #统计两类数据的Gini
                Gini1=2*np.mean(labelYes)*(1-np.mean(labelYes))
                Gini2=2*np.mean(labelNo)*(1-np.mean(labelNo))
                GiniFeat=(len(labelYes)*Gini1+len(labelNo)*Gini2)/len(label)
                if GiniFeat<=VarGini:
                    VarGini=GiniFeat
                    Var=FeatName
                    VarValue=FeatValue
                    VarDiscreate=FeatDiscreate
    if VarDiscreate:
        Data1=Data[Data[Var]==VarValue]
        Data2=Data[Data[Var]!=VarValue]
        label1=label[Data1.index]
        label2=label[Data2.index]
    else:
        Data1=Data[Data[Var]<=VarValue]
        Data2=Data[Data[Var]>VarValue]
        label1=label[Data1.index]
        label2=label[Data2.index]
    #子树为空将其置为叶子节点
    if (len(label1)==0):
        root.LeftChild=TREE()
        root.LeftChild.leave=True
        root.LeftChild.label=label.value_counts().index[0]
    else:
        root.LeftChild=CART(Data1,label1,nmin)
    
    if (len(label2)==0):
        root.RightChild=TREE()
        root.RightChild.leave=True
        root.RightChild.label=label.value_counts().index[0]
    else:
        root.RightChild=CART(Data2,label2,nmin)    
        
    root.VarDiscreate=VarDiscreate
    root.Var=Var
    root.VarValue=VarValue
                
    return root

# %%
class RF:
    def __init__(self,nmin,Featnum,treenum) -> None:
        self.nmin=nmin
        self.Featnum=Featnum
        self.treenum=treenum
        self.forest=[]
    def train(self,data,label):
        self.forest=[]
        oob=[]
        for _ in range(self.treenum):
            #有放回地抽取样本构成训练集合
            trainidx=random.choices(range(len(label)),k=len(label))
            traindata=data.iloc[trainidx]
            a=random.sample(range(len(traindata.columns)),self.Featnum)
            traindata=traindata.iloc[:,a].reset_index(drop=True)
            trainlabel=label.iloc[trainidx].reset_index(drop=True)
            tree = CART(traindata,trainlabel,self.nmin)
            self.forest.append(tree)
            
            trainidx_unique = list(set(trainidx))
            untrainidx = list(range(len(label)))
            for id in trainidx_unique:
                untrainidx.remove(id)
            untraindata=data.iloc[untrainidx].reset_index(drop=True)
            untrainlabel=label.iloc[untrainidx].reset_index(drop=True)
            pred=[]
            for n in range(len(untraindata)):
                pred.append(tree.predict(untraindata.iloc[n]))
            TP=0
            TN=0
            for idx,real in enumerate(untrainlabel):
                if real==1 and pred[idx]==1:
                    TP+=1
                if real==0 and pred[idx]==0:
                    TN+=1
            acc=(TP+TN)/len(untrainlabel)
            oob.append(1-acc)
        return np.mean(oob)
    
    def predict(self,data,th):
        tmp=[]
        for tree in self.forest:
            tmp.append(tree.predict(data))
        #投票
        return (np.mean(tmp)>th)*1

# %%
if __name__ == '__main__':
    df=pd.read_csv('heart_2020_cleaned.csv')
    _label=df[df.columns[0]]
    label = np.zeros(len(_label))
    for i in range(len(_label)):
        if _label[i] == 'Yes':
            label[i] = 1
    label = pd.DataFrame(label)
    label = label[label.columns[0]]

    # %%
    data=df.drop(columns=df.columns[0])
    divide=0.95
    train_data=data.iloc[:int(divide*len(data))]
    train_label=label.iloc[:int(divide*len(data))]
    train_data_p=train_data[train_label==1]
    train_label_p=train_label[train_label==1]
    train_data_n=train_data[train_label==0]
    train_label_n=train_label[train_label==0]
    p_size=5000
    n_size=5000
    p_index=random.sample(range(len(train_data_p)),p_size)
    train_data_p=train_data_p.iloc[p_index].reset_index(drop=True)
    train_label_p=train_label_p.iloc[p_index].reset_index(drop=True)
    n_index=random.sample(range(len(train_data_n)),n_size)
    train_data_n=train_data_n.iloc[n_index].reset_index(drop=True)
    train_label_n=train_label_n.iloc[n_index].reset_index(drop=True)
    train_data=pd.concat([train_data_p,train_data_n])
    train_label=pd.concat([train_label_p,train_label_n])

    test_data=data.iloc[int(divide*len(data)):]
    test_label=label.iloc[int(divide*len(data)):]

    # %%
    record=np.zeros([7,6,4],dtype=np.float32)
    iternum=5
    nmin=30

    for treenum in range(1,14,2):
        for featnum in range(3,9,1):
            record1=[]
            for m in range(iternum):
                cla=RF(nmin,featnum,treenum)
                oob = cla.train(train_data,train_label)
                pred=[]
                for n in range(len(test_data)):
                    pred.append(cla.predict(test_data.iloc[n],0.5))
                TP=0
                FP=0
                TN=0
                FN=0
                for idx,real in enumerate(test_label):
                    if real==1 and pred[idx]==1:
                        TP+=1
                    if real==0 and pred[idx]==1:
                        FP+=1
                    if real==0 and pred[idx]==0:
                        TN+=1  
                    if real==1 and pred[idx]==0:
                        FN+=1  
                acc=(TP+TN)/(TP+FP+TN+FN)
                recall=TP/(TP+FN)
                prec=TP/(TP+FP)
                record1.append([acc,prec,recall,oob])
            record[int((treenum-1)/2),featnum-3,:] = np.mean(np.array(record1),axis=0)

    record=np.array(record)
    np.save('record.npy', record)
        