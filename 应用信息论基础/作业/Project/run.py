import numpy as np
import matplotlib.pyplot as plt
import time

def _down_up(pro, num, tes):
    length = len(pro)
    flag = 1
    sum_1 = np.zeros(length*(length-1)//2)
    ind_1 = np.zeros([length*(length-1)//2, 2])
    k = -1
    for i in range(length-1):
        for j in range(i+1,length):
            k = k + 1 
            sum_1[k] = pro[i] + pro[j]
            ind_1[k, :] = np.array([i,j])
    sum_2 = np.sort(sum_1)
    ind_2 = np.argsort(sum_1)
    for i in range(len(sum_2)):
        x = int(ind_1[ind_2[i],0])
        y = int(ind_1[ind_2[i],1])
        a = num[x, :]                    
        b = num[y, :]
        a = a[a>0]              
        b = b[b>0]
        if (np.min(a) >= np.max(b) or np.max(a) <= np.min(b) or (np.max(a)-np.min(a) == len(a)-1) or (np.max(b)-np.min(b) == len(b)-1)):
            pro_o = pro
            pro_o[x] = pro_o[x] + pro_o[y]
            pro_o = np.delete(pro_o, y)
            num_o = num
            num_o[x,:] = np.concatenate([a,b,np.zeros(6-len(a)-len(b))])
            num_o = np.delete(num_o, y, axis=0)
            tes_o = tes
            tes_o[a-1] = tes_o[a-1]+1
            tes_o[b-1] = tes_o[b-1]+1
            flag = 0                                  
            break
  
    if flag == 1:
        pro_o = pro
        num_o = num
        tes_o = tes
        
    return pro_o, num_o, tes_o, flag


def down_up(prob):
    pro_o = prob
    num = 0 * np.ones([6,6], dtype=np.int8)
    num[:,0] = np.arange(1,7)
    tes = np.zeros(6)
    
    for i in range(5):
        [pro_o,num,tes,flag] = _down_up(pro_o, num, tes)
        if flag == 1:
            break
        
    if flag == 1:
        ave_c = -1
    else:
        ave_c = np.sum(prob * tes)
        
    return ave_c


def _up_down(pro, num, tes):
    
    length = len(pro)

    srart = 0
    over = 0
    dif = np.abs(np.sum(pro[srart:over+1])-np.sum(pro)/2)
    for i in range(length): 
        for j in range(i, length):
            dif_new = np.abs(np.sum(pro[i:j+1])-np.sum(pro)/2)
            if dif_new <= dif:
                dif = dif_new
                start = i
                over  = j



    pro_01 = pro[start:over+1]
    num_01 = num[start:over+1]
    tes_01 = tes[start:over+1] + 1

    if start == 0:
        pro_02 = pro[over+1:]
        num_02 = num[over+1:]
        tes_02 = tes[over+1:] + 1
    elif over == length-1:
        pro_02 = pro[:start]
        num_02 = num[:start]
        tes_02 = tes[:start] + 1
    else:
        pro_02 = np.concatenate([pro[:start],pro[over+1:]])
        num_02 = np.concatenate([num[:start],num[over+1:]])
        tes_02 = np.concatenate([tes[:start],tes[over+1:]]) + 1

    if over-start+1 >= 3:
        pro_1 = pro_01
        tes_1 = tes_01
        num_1 = num_01
        pro_2 = pro_02
        tes_2 = tes_02
        num_2 = num_02
    else:
        pro_1 = pro_02
        tes_1 = tes_02
        num_1 = num_02
        pro_2 = pro_01
        tes_2 = tes_01
        num_2 = num_01

    return pro_1, pro_2, num_1, num_2, tes_1, tes_2

def up_down(pro):

    ave_c = 0                       
    min_c = 0                     

    length = len(pro)
    num = np.arange(length)              
    tes_m = np.zeros(length)          
    tes_o = np.zeros(length)

    [pro_1,pro_2,num_1,num_2,tes_1,tes_2] = _up_down(pro, num, tes_m)

    if len(pro_1) == 5:
        [pro_11,pro_12,num_11,num_12,tes_11,tes_12] = _up_down(pro_1,num_1,tes_1)
        if len(pro_11) == 4:
            [pro_21,pro_22,num_21,num_22,tes_21,tes_22] = _up_down(pro_11,num_11,tes_11)
            if len(pro_21) == 3:
                [_,_,num_31,num_32,tes_31,tes_32] = _up_down(pro_21,num_21,tes_21)
                tes_o[num_31] = tes_31 + len(num_31) - 1
                tes_o[num_32] = tes_32 + len(num_32) - 1
                tes_o[num_22] = tes_22
            else:
                tes_o[num_11] = tes_11 + 2
            tes_o[num_12] = tes_12
        else:
            [_,_,num_21,num_22,tes_21,tes_22] = _up_down(pro_11,num_11,tes_11)
            tes_o[num_21] = tes_21 + len(num_21)-1
            tes_o[num_22] = tes_22 + len(num_22)-1
            tes_o[num_12] = tes_12 + 1
        tes_o[num_2] = tes_2
    elif len(pro_1) == 4:
        [pro_11,pro_12,num_11,num_12,tes_11,tes_12] = _up_down(pro_1,num_1,tes_1)
        if len(pro_11) == 3:
            [_,_,num_21,num_22,tes_21,tes_22] = _up_down(pro_11,num_11,tes_11)
            tes_o[num_21] = tes_21 + len(num_21)-1
            tes_o[num_22] = tes_22 + len(num_22)-1
            tes_o[num_12] = tes_12
        else:
            tes_o[num_1] = tes_1 + 2
        tes_o[num_2] = tes_2 + 1
    elif len(pro_1) == 3:
        [_,_,num_11,num_12,tes_11,tes_12] = _up_down(pro_1,num_1,tes_1)
        [_,_,num_21,num_22,tes_21,tes_22] = _up_down(pro_2,num_2,tes_2)
        tes_o[num_11] = tes_11 + len(num_11)-1
        tes_o[num_12] = tes_12 + len(num_12)-1
        tes_o[num_21] = tes_21 + len(num_21)-1
        tes_o[num_22] = tes_22 + len(num_22)-1
    
    ave_c = np.sum(pro * tes_o)

    return ave_c


if __name__ == '__main__':
    N = 100000
    pro = np.random.random([N,6])
    y = np.sum(pro, axis=1).reshape(-1,1)
    a = np.kron(y, np.ones([1,6]))
    pro = pro / np.kron(y, np.ones([1,6]))
     
    ave_1 = np.zeros(N)          
    ave_2 = np.zeros(N)       

    # both
    # for i in range(N):
    #     if ave_1[i] != -1:
    #         ave_2[i] = up_down(pro[i,:])
    # print(np.mean(ave_1[ave_1>0]), np.mean(ave_2[ave_2>0]))
    # print(np.var(ave_1[ave_1>0]), np.var(ave_2[ave_2>0]))
    # print(np.mean(ave_2), np.var(ave_2))
    # plt.figure()
    # plt.hist(ave_1,bins=100,color='blue',edgecolor='black')
    # plt.hist(ave_2,bins=100,color='orange',edgecolor='black')
    # plt.xlim([1.5,3.5])
    # plt.ylim([0,30000])
    # plt.legend(['down up','up down'])
    # plt.xlabel('interval')
    # plt.ylabel('frequency')
    # plt.show()

    # up down
    # for i in range(N):
    #     ave_2[i] = up_down(pro[i,:])
    # print(np.mean(ave_2), np.var(ave_2), np.min(ave_2), np.max(ave_2))
    # plt.figure()
    # plt.hist(ave_2,bins=100,color='orange',edgecolor='black')
    # plt.xlim([1,3])
    # plt.ylim([0,8000])
    # plt.legend(['up down'])
    # plt.xlabel('interval')
    # plt.ylabel('frequency')
    # plt.show()

    # down up
    # for i in range(N):
    #     ave_1[i] = down_up(pro[i,:])
    # print(np.mean(ave_1[ave_1>0]), np.var(ave_1[ave_1>0]), len(ave_1[ave_1>0])/N, np.min(ave_1[ave_1>0]), np.max(ave_1[ave_1>0]))
    # plt.figure()
    # plt.hist(ave_1,bins=100,color='orange',edgecolor='black')
    # plt.xlim([1.5,3.5])
    # plt.ylim([0,3000])
    # plt.legend(['down up'])
    # plt.xlabel('interval')
    # plt.ylabel('frequency')
    # plt.show()
    
    # test time
    s = time.time()
    for i in range(N):
        ave_1[i] = down_up(pro[i,:])
    print(time.time()-s)
    s = time.time()
    for i in range(N):
        ave_2[i] = up_down(pro[i,:])
    print(time.time()-s)