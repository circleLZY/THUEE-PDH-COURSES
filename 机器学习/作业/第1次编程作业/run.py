# %% [markdown]
# Loading modules...
# This task needs numpy, pandas and matplotlib
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Loading data... \
# Gender and Nationality should be coded

# %%
class dataLoader:
    def __init__(self, data) -> None:
        self.data = data
    
    def code_country(self):
        country_dict = {'France':0, 'Spain':1, 'Germany':2}
        for i in range(len(self.data['country'])):
            self.data['country'][i] = country_dict[self.data['country'][i]]
            
    def code_gender(self):
        gender_dict = {'Female':0, 'Male':1}
        for i in range(len(self.data['gender'])):
            self.data['gender'][i] = gender_dict[self.data['gender'][i]]
    
    def regularize(self, label='norm'):
        features = self.data.columns[:-1]
        if label == 'none':
            self.data = self.data
        elif label == 'norm':
            for feature in features:
                mu = self.data[feature].mean()
                sigma = self.data[feature].var()
                self.data[feature] = (self.data[feature] - mu) / sigma
        elif label == 'unif':
            for feature in features:
                self.data[feature] = (self.data[feature] - self.data[feature].min()) / (self.data[feature].max() - self.data[feature].min())
        else:
            raise(NotImplementedError)
    
    def preprocess(self, country_coded=True, gender_coded=True, label='norm'):
        self.data.drop(columns=['customer_id'], axis=1, inplace=True)
        
        if country_coded == False:
            self.data.drop(columns=['country'], axis=1, inplace=True)
        else:
            self.code_country()
            
        if gender_coded == False:
            self.data.drop(columns=['gender'], axis=1, inplace=True)
        else:
            self.code_gender()
        
        self.regularize(label)
    

# %%
class logisticTrain:
    def __init__(self, data, train_set=7000, valid_set=8000, test_set=10000, lr=0.001, Lambda=0.001, batch=100, epochs=100, T=0.001) -> None:
        self.data = np.array(data, dtype=np.float32)
        
        self.x = np.column_stack((np.ones((len(self.data[:,0]),1)), self.data[:,:-1]))
        self.train_x = self.x[:train_set,:]
        self.valid_x = self.x[train_set:valid_set,:]
        self.test_x = self.x[valid_set:test_set,:]
        
        self.t = np.array(self.data[:,-1])
        self.train_t = self.t[:train_set]
        self.valid_t = self.t[train_set:valid_set]
        self.test_t = self.t[valid_set:test_set]
        
        self.w = np.zeros((len(self.x[0,:]), 1))
        self.lr =  lr
        self.Lambda = Lambda
        self.batch = batch
        self.epochs = epochs
        self.T = T
        self.loss = []
        self.precision = []
        self.recall = []
        self.accuracy = []
        # self.LR = []
        
        
    def train(self):
        its = len(self.train_t) // self.batch
        for epoch in range(self.epochs):
            lr = self.lr * np.exp(-self.T*epoch)
            # self.LR.append(lr)
            # lr = self.lr
            # self.LR.append(lr)
            for it in range(its):
                x = self.train_x[it*self.batch:(it+1)*self.batch, :] 
                t = self.train_t[it*self.batch:(it+1)*self.batch].reshape(-1,1)
                y = 1. / (1. + np.exp(-np.dot(x, self.w)))
                loss = -np.sum(t*np.log(y) + (1-t)*np.log(1-y)) + 0.5*self.Lambda*np.dot(self.w.T, self.w)
                # loss = -np.sum(t*np.log(y) + (1-t)*np.log(1-y))/self.batch + 0.5*self.Lambda*np.dot(self.w.T, self.w)
                # loss = -np.sum(t*np.log(y) + (1-t)*np.log(1-y))/self.batch
                self.loss.append(loss.item())
                dw = np.dot(x.T, y-t) + self.Lambda*self.w
                # dw = np.dot(x.T, y-t)/self.batch + self.Lambda*self.w
                # dw = np.dot(x.T, y-t)/self.batch 
                self.w = self.w - lr*dw
        
            if len(self.valid_t) > 0:
                y = (np.sign(np.dot(self.valid_x, self.w)) + 1) / 2 
                self.test(y, self.valid_t)
            
            if len(self.test_t) > 0:
                y = (np.sign(np.dot(self.test_x, self.w)) + 1) / 2 
                self.test(y, self.test_t)
    
    
    def regTrain(self):
        its = len(self.train_t) // self.batch
        for epoch in range(self.epochs):
            lr = self.lr * np.exp(-self.T*epoch)
            for it in range(its):
                x = self.train_x[it*self.batch:(it+1)*self.batch, :] 
                t = self.train_t[it*self.batch:(it+1)*self.batch].reshape(-1,1)
                y = np.dot(x, self.w)
                loss = 0.5*(np.sum(np.square(t-y)) + self.Lambda*np.dot(self.w.T, self.w))
                self.loss.append(loss.item())
                dw = np.dot(x.T, y-t) + self.Lambda*self.w
                self.w = self.w - lr*dw
            
            if len(self.valid_t) > 0:
                y = (np.sign(np.dot(self.valid_x, self.w) - 0.5) + 1) / 2 
                self.test(y, self.valid_t)
            
            if len(self.test_t) > 0:
                y = (np.sign(np.dot(self.test_x, self.w) - 0.5) + 1) / 2 
                self.test(y, self.test_t)
    

    def test(self, y, t):
        NTP = 0
        NFP = 0
        NFN = 0
        NTN = 0
        eps = 1e-6
        for i in range(len(t)):
            if t[i] == 0 and y[i] == 0:
                NTN = NTN + 1
            if t[i] == 1 and y[i] == 0:
                NFN = NFN + 1
            if t[i] == 1 and y[i] == 1:
                NTP = NTP + 1
            if t[i] == 0 and y[i] == 1:
                NFP = NFP + 1
        self.precision.append(NTP / (NTP + NFP + eps))
        self.recall.append(NTP / (NTP + NFN + eps))
        self.accuracy.append((NTP + NTN) / len(t))
       
        
    def draw(self, title):
        plt.figure('loss')
        plt.plot(self.loss)
        plt.show()
        
        plt.figure(title)
        plt.plot(self.accuracy)
        plt.plot(self.precision)
        plt.plot(self.recall)
        plt.legend(['accuracy', 'precision', 'recall'])
        plt.show()


if __name__ == '__main__':
    # %%
    data = pd.read_csv('Bank Customer Churn Prediction.csv')
    data_loader = dataLoader(data)
    # data_loader.data

    # %% [markdown]
    # Test if gender and nationality can influence factors

    # %%
    # country_list = []
    # for i in  data_loader.data['country']:
    #     if i not in country_list:
    #         country_list.append(i)
    # country_list

    # %%
    # country_dict = {'France':{'balance':[], 'cnt':0, 'mean':0, 'var':0}, 
    #                 'Spain':{'balance':[], 'cnt':0, 'mean':0, 'var':0}, 
    #                 'Germany':{'balance':[], 'cnt':0, 'mean':0, 'var':0}}

    # for i in range(len(data_loader.data)):
    #     country_dict[data_loader.data['country'][i]]['balance'].append(data_loader.data['balance'][i])
    #     country_dict[data_loader.data['country'][i]]['cnt'] += 1
    # for i in country_list:
    #     country_dict[i]['mean'] = np.mean(country_dict[i]['balance'])
    #     country_dict[i]['var'] = np.var(country_dict[i]['balance'])
    #     del(country_dict[i]['balance'])
    # country_dict

    # %%
    # gender_list = []
    # for i in  data_loader.data['gender']:
    #     if i not in gender_list:
    #         gender_list.append(i)
    # gender_list

    # %%
    # gender_dict = {'Male':{'estimated_salary':[], 'cnt':0, 'mean':0, 'var':0}, 
    #                'Female':{'estimated_salary':[], 'cnt':0, 'mean':0, 'var':0}}

    # for i in range(len(data_loader.data)):
    #     gender_dict[data_loader.data['gender'][i]]['estimated_salary'].append(data_loader.data['estimated_salary'][i])
    #     gender_dict[data_loader.data['gender'][i]]['cnt'] += 1
    # for i in gender_list:
    #     gender_dict[i]['mean'] = np.mean(gender_dict[i]['estimated_salary'])
    #     gender_dict[i]['var'] = np.var(gender_dict[i]['estimated_salary'])
    #     del(gender_dict[i]['estimated_salary'])
    # gender_dict

    # %%
    data_loader.preprocess(country_coded=True, gender_coded=True, label='norm')
    data = data_loader.data

    # %%
    # data = np.array(data_loader.data, dtype=np.float32)
    # np.save('data.npy', data)

    # %%
    # data = np.load('data.npy')

    # %%
    trainer = logisticTrain(data, 
                            train_set=8000, 
                            valid_set=8000, 
                            test_set=10000, 
                            lr=0.00001,
                            Lambda=0.001, 
                            batch=8000, 
                            epochs=10000,
                            T=0.0001)

    # %%
    trainer.train()
    trainer.draw('test')

    # %%
    trainer.regTrain()
    trainer.draw('test') 

    # %% [markdown]
    # 


