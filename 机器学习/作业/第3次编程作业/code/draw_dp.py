import numpy as np
import matplotlib.pyplot as plt
import pickle

for i in range(1,10,1):
    plt.figure()
    with open('drop'+str(i)+'.pkl','rb') as f:
        drop=pickle.load(f)
    train_loss_drop=drop[0]
    test_loss_drop=drop[1]
    plt.plot(train_loss_drop)
    plt.plot(test_loss_drop)
    plt.legend(['train_loss_with_drop_BN','test_loss_with_drop_BN'])
    plt.title('loss with dropout 0.'+str(i))
    plt.savefig('loss_'+str(i)+'.png')
    print(i,'train_acc_top1:',drop[2][-1],'train_acc_top5:',drop[3][-1],'test_acc_top1:',drop[4][-1],'test_acc_top5:',drop[5][-1])

results=['train_loss','test_loss','train_acc_top1','train_acc_top5','test_acc_top1','test_acc_top5']
for j in range(6):
    plt.figure()
    for i in range(1,10,1):
        with open('drop'+str(i)+'.pkl','rb') as f:
            drop=pickle.load(f)
        plt.plot(drop[j])
    plt.legend(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
    plt.title(results[j])
    plt.savefig(results[j]+'.png')