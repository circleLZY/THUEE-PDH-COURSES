clear;
close all;
clc;

N     = 20;                   % 实验次数
SNR   = 20;                   % 信噪比
sigma = 10^(-SNR/20);
M     = 11;                   % 均衡器阶数
mu    = 0.2;                  % 步长

e0 = [];
w0 = [];
for a = 1:20
    x = 2*rand(1,1e3)-1;                % 随机信号生成
    x(x>0) =  1;
    x(x<0) = -1;
    
    s = conv(x,[0.3,0.9,0.3]);          % 接收信号
    w = normrnd(0,sigma,[1,length(s)]); % 噪声
    r = s+w;                            % 接收信号
    
    w = zeros(1,M);
    for i = 301:800
        e(i-300)  = x(i-7) - sum(w.*r(i-M+1:i));
        xx(i-300) = sum(r(i-M+1:i).*r(i-M+1:i));
        w = w + mu*r(i-M+1:i)*e(i-300)/xx(i-300);
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





