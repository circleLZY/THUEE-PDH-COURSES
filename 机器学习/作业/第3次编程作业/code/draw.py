import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('drop.pkl','rb') as f:
    drop=pickle.load(f)
with open('none.pkl','rb') as f:
    no=pickle.load(f)
train_loss_drop=drop[0]
test_loss_drop=drop[1]
train_loss=no[0]
test_loss=no[1]
plt.plot(train_loss_drop)
plt.plot(test_loss_drop)
plt.legend(['train_loss_with_drop_BN','test_loss_with_drop_BN'])
plt.savefig('loss_BN')
plt.figure()
plt.plot(train_loss)
plt.plot(test_loss)
plt.legend(['train_loss_without_drop_BN','test_loss_without_drop_BN'])
plt.savefig('loss_out')
print('with dropout and bn  train_acc_top1:',drop[2][-1],'train_acc_top5:',drop[3][-1],'test_acc_top1:',drop[4][-1],'test_acc_top5:',drop[5][-1])
print('without dropout or bn  train_acc_top1:',no[2][-1],'train_acc_top5:',no[3][-1],'test_acc_top1:',no[4][-1],'test_acc_top5:',no[5][-1])
