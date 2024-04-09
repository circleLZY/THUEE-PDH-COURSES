clear all;
close all;
clc;


%********** 准备工作 **********%
% 参数配置
N = 5e4;           % 样本总量
a_sigma =  2;      % 加速度噪声方差
w_sigma = 5;       % 角动量噪声方差
T = N/250;          % 缓变信号周期为20，快变信号周期为4
a_a = 1;           % 加速度的变化幅度
w_a = 100;         % 角速度的变化幅度

% 生成运动信号s
s_m = zeros(6,N);
a0 = zeros(1,T);
a0(1, 1:T/4) = a_a*(1:T/4)/(T/4);
a0(1, T/4+1:3*T/4) = -a_a*((1:T/2)/(T/4)-1);
a0(1, 3*T/4+1:T) = a0(1, 1:T/4) - a_a;
w0 = zeros(1,T);
w0(1, 1:T/4) = w_a*(1:T/4)/(T/4);
w0(1, T/4+1:3*T/4) = -w_a*((1:T/2)/(T/4)-1);
w0(1, 3*T/4+1:T) = w0(1, 1:T/4) - w_a;
for id = 1:N/T
    s_m(1, (id-1)*T+1:id*T) = a0;
    s_m(2, (id-1)*T+1:id*T) = a0;
    s_m(3, (id-1)*T+1:id*T) = a0;
    s_m(4, (id-1)*T+1:id*T) = w0;
    s_m(5, (id-1)*T+1:id*T) = w0;
    s_m(6, (id-1)*T+1:id*T) = w0;
end

figure;
hold on;
t = 1:10*T;
subplot(2,1,1),plot(t,s_m(1,1:length(t))),title('加速度信号');
subplot(2,1,2),plot(t,s_m(4,1:length(t))),title('角速度信号');

% 生成静止信号
s_s = zeros(6,N);
s_s(1:3,:) = rand([3,N]);
s_s(1:3,:) = sqrt(10*s_s(1:3,:).*s_s(1:3,:) ./ sum(s_s(1:3,:).*s_s(1:3,:)));


%********** 检测判决 **********%
% 生成接收信号
y_m = s_m + [normrnd(0,a_sigma,[3,N]);normrnd(0,w_sigma,[3,N])];
y_s = s_s + [normrnd(0,a_sigma,[3,N]);normrnd(0,w_sigma,[3,N])];

M    = [3,5,10,20];    % 检测窗长(/100 s)
thre = [1e2:2e2:2e3];  % 检测阈值
PFA  = zeros(length(M),length(thre));
PD   = zeros(length(M),length(thre));

figure,hold on;
for i = 1:length(M)
    for j = 1:length(thre)
        calcul_moving = zeros(1,N-M(i)+1);
        result_moving = zeros(1,N-M(i)+1);       % 0：运动；1：静止；
        calcul_statio = zeros(1,N-M(i)+1);      
        result_statio = zeros(1,N-M(i)+1);       % 0：运动；1：静止；
        
        
        % 检测判决
        for k = 1:N-M(i)+1
            calcul_moving(k) = mean(sum(y_m(4:6,k:k+M(i)-1) .* y_m(4:6,k:k+M(i)-1)));
            if (calcul_moving(k) <= thre(j))
                result_moving(k) = 1;
            else
                result_moving(k) = 0;
            end
            calcul_statio(k) = mean(sum(y_s(4:6,k:k+M(i)-1) .* y_s(4:6,k:k+M(i)-1)));  
            if (calcul_statio(k) <= thre(j))
                result_statio(k) = 1;
            else
                result_statio(k) = 0;
            end
        end
        
        PFA(i,j) = sum(result_moving == 1)/(N-M(i)+1); 
        PD (i,j) = sum(result_statio == 1)/(N-M(i)+1);
    end
    plot(PFA(i,:),PD(i,:),'-*');
end
legend('M = 3','M = 5','M = 10','M = 20');
xlabel('PFA');
ylabel('PD');
title(['快变信号，noise sigma = ',num2str(w_sigma)]);

