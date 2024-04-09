clear;
close all;
clc;

N     = 20;                   % 实验次数
SNR   = 10;                   % 信噪比
sigma = 10^(-SNR/20);

M     = 11;                   % 均衡器阶数
lam   = 0.8;                  % 忘却因子
del   = 1e-3;                 % P初值 

e0 = [];
w0 = [];
for a = 1:N
    x = 2*rand(1,1e3)-1;                % 随机信号生成
    x(x>0) =  1;
    x(x<0) = -1;
    
    s = conv(x,[0.3,0.9,0.3]);          % 接收信号
    w = normrnd(0,sigma,[1,length(s)]); % 噪声
    r = s+w;                            % 接收信号
    
    P = eye(M)/del;
    w = zeros(1,M);
    for i = 301:800
        rn = r(i-M+1:i);
        kn = (P*rn'/lam)/(1+rn*P*rn'/lam);     % 增益向量
        en = x(i-7) - sum(w.*rn);              % 前验估计误差
    
        w = w+kn'*en;                           % 更新权系数向量 
        P = P/lam -kn*rn*P/lam;                % 系数逆矩阵
        e(i-300) = en;
    end

    e0 = [e0;e.^2];
    w0 = [w0;w];
end
figure,hold on;
subplot(1,2,1),plot(e0(1,:));
title('1次实验误差平方曲线');
xlabel('训练序列序号');
ylabel('误差平方');
subplot(1,2,2),plot(mean(e0));
title('20次实验平均误差平方曲线');
xlabel('训练序列序号');
ylabel('误差平方');

fprintf('%3d\n',w0(1,:));
