clear all;
close all;
clc;

load data2.mat;
signal_use = data2';
signal_len = length(signal_use(1,:));

M = 10;            % 检测窗长
th = 0.1;

T = zeros(1,signal_len);    % 统计量
H = zeros(1,signal_len);    % 判定结果
for i = 1:signal_len-M+1
    T(i) = mean(sum(signal_use(4:6,i:i+M-1) .* signal_use(4:6,i:i+M-1)));
    if (T(i) < th)
        H(i) = 1;
    else
        H(i) = 0;
    end
end

figure;
grid on;
t = (1:length(T))/100;
semilogy(t,T);
hold on;
plot(t,th*ones(1,signal_len),'k'),xlabel('时间'),ylabel('统计量数值');
figure;
plot(t,H),xlabel('时间'),ylabel('判决结果');










