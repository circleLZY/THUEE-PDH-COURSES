clear all,close all,clc;

A = 10;
r = 1.05;
N = 5e2;


x   = zeros(1,N+1);
A_e = zeros(1,N+1);             % 估计结果
K   = zeros(1,N+1);             % 增益因子
v_e = zeros(1,N+1);             % 方差序列

x(1) = A + normrnd(0,sqrt(r^0));
A_e(1) = x(1);
v_e(1) = 1;
for i = 1:N
    x(i+1) = A + normrnd(0,sqrt(r^i));
    K(i+1) = (1/r^i) / (1/(v_e(i))+(1/r^i));
    A_e(i+1) = A_e(i) + K(i+1)*(x(i+1)-A_e(i));
    v_e(i+1) = (1-K(i+1))*v_e(i);
end

figure,hold on;
subplot(1,3,1),plot(A_e),title('A的估计结果','FontSize',16);
subplot(1,3,2),plot(K  ),title('增益因子','FontSize',16);
subplot(1,3,3),plot(v_e),title('估计结果方差','FontSize',16);
set (gcf,'Position', [100,100,1000,300]);
