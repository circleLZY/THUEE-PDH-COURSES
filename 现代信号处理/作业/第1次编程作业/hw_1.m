clear; close all; clc;
digits(50);

%********** load data **********%
load("vel_data.mat")
load("pos_data.mat")
sigma_a = 1;
sigma_p = 36;
sigma_v = 1;
T = 0.005;
N = length(pos_e);
v = [normrnd(0, T^2*sigma_a/2, [1,N]); normrnd(0, T*sigma_a, [1,N])];
w  = normrnd(0, sigma_p, [1,N]);    % 观测噪声

%********** 卡尔曼滤波 **********%
% 第一问
I  = [1,0;0,1];
F  = [1,T;0,1];
C  = [1,0];
Q1 = [T^4/4,T^3/2;T^3/2,T^2]*sigma_a;
Q2 = sigma_p;

x_e      = zeros(2,N+1);
x_e(:,1) = [10,10];
x_n      = zeros(2,N+1);
x_n(:,1) = [10,10];
K_e      = [1000,0;0,1000];
K_n      = [1000,0;0,1000];

for i = 1:N
    % east
    x_e_tmp    = F*x_e(:,i);
    a          = pos_e(i) - C*x_e_tmp;
    K_e_tmp    = F*K_e*F' + Q1;
    R          = C*K_e_tmp*C' + Q2;
    G_f        = K_e_tmp*C'*R^-1;
    x_e(:,i+1) = x_e_tmp + G_f*a;
    K_e        = (I-G_f*C)*K_e_tmp;
    % north
    x_n_tmp    = F*x_n(:,i);
    a          = pos_n(i) - C*x_n_tmp;
    K_n_tmp    = F*K_n*F' + Q1;
    R          = C*K_n_tmp*C' + Q2;
    G_f        = K_n_tmp*C'*R^-1;
    x_n(:,i+1) = x_n_tmp + G_f*a;
    K_n        = (I-G_f*C)*K_n_tmp;
end

figure,hold on;
subplot(2,1,1),hold on,plot(pos_e);plot(x_e(1,1:end-1));
legend('real data','Kalman filter');
title('east position of the target');
subplot(2,1,2),hold on,plot(vel_e);plot(x_e(2,1:end-1));
legend('real data',' Kalman filter');
title('east velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_e(1,1:end-1)-pos_e);
title('residuals of position');
subplot(2,1,2),plot(x_e(2,1:end-1)-vel_e);
title('residuals of velocity');

disp('east position mse: '), disp(mse(x_e(1,1:end-1)-pos_e));
disp('east velocity mse: '), disp(mse(x_e(2,1:end-1)-vel_e));

figure,hold on;
subplot(2,1,1),hold on,plot(pos_n);plot(x_n(1,1:end-1));
legend('real data','Kalman filter');
title('north position of the target');
subplot(2,1,2),hold on,plot(vel_n);plot(x_n(2,1:end-1));
legend('real data',' Kalman filter');
title('north velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_n(1,1:end-1)-pos_n);
title('residuals of position');
subplot(2,1,2),plot(x_n(2,1:end-1)-vel_n);
title('residuals of velocity');

disp('north position mse: '), disp(mse(x_n(1,1:end-1)-pos_n));
disp('north velocity mse: '), disp(mse(x_n(2,1:end-1)-vel_n));

figure,hold on;
plot(pos_e, pos_n, LineWidth=2);
plot(x_e(1,1:end-1), x_n(1,1:end-1), LineWidth=2);
legend('real data',' Kalman filter');

% 第二问
C  = [1,0;0,1];
Q2 = [sigma_p, sigma_v];
x_e      = zeros(2,N+1);
x_e(:,1) = [10,10];
x_n      = zeros(2,N+1);
x_n(:,1) = [10,10];
K_e      = [1000,0;0,1000];
K_n      = [1000,0;0,1000];
y_e      = [pos_e; vel_e];
y_n      = [pos_n; vel_n];

for i = 1:N
    % east
    x_e_tmp    = F*x_e(:,i);
    a          = y_e(:,i) - C*x_e_tmp;
    K_e_tmp    = F*K_e*F' + Q1;
    R          = C*K_e_tmp*C' + Q2;
    G_f        = K_e_tmp*C'*R^-1;
    x_e(:,i+1) = x_e_tmp + G_f*a;
    K_e        = (I-G_f*C)*K_e_tmp;
    % north
    x_n_tmp    = F*x_n(:,i);
    a          = y_n(:,i) - C*x_n_tmp;
    K_n_tmp    = F*K_n*F' + Q1;
    R          = C*K_n_tmp*C' + Q2;
    G_f        = K_n_tmp*C'*R^-1;
    x_n(:,i+1) = x_n_tmp + G_f*a;
    K_n        = (I-G_f*C)*K_n_tmp;
end

figure,hold on;
subplot(2,1,1),hold on,plot(pos_e);plot(x_e(1,1:end-1));
legend('real data','Kalman filter');
title('east position of the target');
subplot(2,1,2),hold on,plot(vel_e);plot(x_e(2,1:end-1));
legend('real data',' Kalman filter');
title('east velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_e(1,1:end-1)-pos_e);
title('residuals of position');
subplot(2,1,2),plot(x_e(2,1:end-1)-vel_e);
title('residuals of velocity');

disp('east position mse: '), disp(mse(x_e(1,1:end-1)-pos_e));
disp('east velocity mse: '), disp(mse(x_e(2,1:end-1)-vel_e));

figure,hold on;
subplot(2,1,1),hold on,plot(pos_n);plot(x_n(1,1:end-1));
legend('real data','Kalman filter');
title('north position of the target');
subplot(2,1,2),hold on,plot(vel_n);plot(x_n(2,1:end-1));
legend('real data',' Kalman filter');
title('north velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_n(1,1:end-1)-pos_n);
title('residuals of position');
subplot(2,1,2),plot(x_n(2,1:end-1)-vel_n);
title('residuals of velocity');

disp('north position mse: '), disp(mse(x_n(1,1:end-1)-pos_n));
disp('north velocity mse: '), disp(mse(x_n(2,1:end-1)-vel_n));

figure,hold on;
plot(pos_e, pos_n, LineWidth=2);
plot(x_e(1,1:end-1), x_n(1,1:end-1), LineWidth=2);
legend('real data',' Kalman filter');

% 第三问
I  = [1,0,0;0,1,0;0,0,1];
F  = [1,T,T^2/2;0,1,T;0,0,1];
C  = [1,0,0];
Q1 = [T^4/4,T^3/2,T^2/2;T^3/2,T^2,T;T^2/2,T,1]*sigma_a;
Q2 = sigma_p;


x_e      = zeros(3,N+1);
x_e(:,1) = [10,10,0];
x_n      = zeros(3,N+1);
x_n(:,1) = [10,10,0];
K_e      = [1000,0,0;0,1000,0;0,0,1000];
K_n      = [1000,0,0;0,1000,0;0,0,1000];

for i = 1:N
    % east
    x_e_tmp    = F*x_e(:,i);
    a          = pos_e(i) - C*x_e_tmp;
    K_e_tmp    = F*K_e*F' + Q1;
    R          = C*K_e_tmp*C' + Q2;
    G_f        = K_e_tmp*C'*R^-1;
    x_e(:,i+1) = x_e_tmp + G_f*a;
    K_e        = (I-G_f*C)*K_e_tmp;
    % north
    x_n_tmp    = F*x_n(:,i);
    a          = pos_n(i) - C*x_n_tmp;
    K_n_tmp    = F*K_n*F' + Q1;
    R          = C*K_n_tmp*C' + Q2;
    G_f        = K_n_tmp*C'*R^-1;
    x_n(:,i+1) = x_n_tmp + G_f*a;
    K_n        = (I-G_f*C)*K_n_tmp;
end

figure,hold on;
subplot(2,1,1),hold on,plot(pos_e);plot(x_e(1,1:end-1));
legend('real data','Kalman filter');
title('east position of the target');
subplot(2,1,2),hold on,plot(vel_e);plot(x_e(2,1:end-1));
legend('real data',' Kalman filter');
title('east velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_e(1,1:end-1)-pos_e);
title('residuals of position');
subplot(2,1,2),plot(x_e(2,1:end-1)-vel_e);
title('residuals of velocity');

disp('east position mse: '), disp(mse(x_e(1,1:end-1)-pos_e));
disp('east velocity mse: '), disp(mse(x_e(2,1:end-1)-vel_e));

figure,hold on;
subplot(2,1,1),hold on,plot(pos_n);plot(x_n(1,1:end-1));
legend('real data','Kalman filter');
title('north position of the target');
subplot(2,1,2),hold on,plot(vel_n);plot(x_n(2,1:end-1));
legend('real data',' Kalman filter');
title('north velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_n(1,1:end-1)-pos_n);
title('residuals of position');
subplot(2,1,2),plot(x_n(2,1:end-1)-vel_n);
title('residuals of velocity');

disp('north position mse: '), disp(mse(x_n(1,1:end-1)-pos_n));
disp('north velocity mse: '), disp(mse(x_n(2,1:end-1)-vel_n));

figure,hold on;
plot(pos_e, pos_n, LineWidth=2);
plot(x_e(1,1:end-1), x_n(1,1:end-1), LineWidth=2);
legend('real data',' Kalman filter');

% 第四问
C  = [1,0,0;0,1,0];
Q2 = [sigma_p, sigma_v];
x_e      = zeros(3,N+1);
x_e(:,1) = [10,10,0];
x_n      = zeros(3,N+1);
x_n(:,1) = [10,10,0];
K_e      = [1000,0,0;0,1000,0;0,0,1000];
K_n      = [1000,0,0;0,1000,0;0,0,1000];
y_e      = [pos_e; vel_e];
y_n      = [pos_n; vel_n];

for i = 1:N
    % east
    x_e_tmp    = F*x_e(:,i);
    a          = y_e(:,i) - C*x_e_tmp;
    K_e_tmp    = F*K_e*F' + Q1;
    R          = C*K_e_tmp*C' + Q2;
    G_f        = K_e_tmp*C'*R^-1;
    x_e(:,i+1) = x_e_tmp + G_f*a;
    K_e        = (I-G_f*C)*K_e_tmp;
    % north
    x_n_tmp    = F*x_n(:,i);
    a          = y_n(:,i) - C*x_n_tmp;
    K_n_tmp    = F*K_n*F' + Q1;
    R          = C*K_n_tmp*C' + Q2;
    G_f        = K_n_tmp*C'*R^-1;
    x_n(:,i+1) = x_n_tmp + G_f*a;
    K_n        = (I-G_f*C)*K_n_tmp;
end

figure,hold on;
subplot(2,1,1),hold on,plot(pos_e);plot(x_e(1,1:end-1));
legend('real data','Kalman filter');
title('east position of the target');
subplot(2,1,2),hold on,plot(vel_e);plot(x_e(2,1:end-1));
legend('real data',' Kalman filter');
title('east velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_e(1,1:end-1)-pos_e);
title('residuals of position');
subplot(2,1,2),plot(x_e(2,1:end-1)-vel_e);
title('residuals of velocity');

disp('east position mse: '), disp(mse(x_e(1,1:end-1)-pos_e));
disp('east velocity mse: '), disp(mse(x_e(2,1:end-1)-vel_e));

figure,hold on;
subplot(2,1,1),hold on,plot(pos_n);plot(x_n(1,1:end-1));
legend('real data','Kalman filter');
title('north position of the target');
subplot(2,1,2),hold on,plot(vel_n);plot(x_n(2,1:end-1));
legend('real data',' Kalman filter');
title('north velocity of the target');

figure,hold on;
subplot(2,1,1),plot(x_n(1,1:end-1)-pos_n);
title('residuals of position');
subplot(2,1,2),plot(x_n(2,1:end-1)-vel_n);
title('residuals of velocity');

disp('north position mse: '), disp(mse(x_n(1,1:end-1)-pos_n));
disp('north velocity mse: '), disp(mse(x_n(2,1:end-1)-vel_n));

figure,hold on;
plot(pos_e, pos_n, LineWidth=2);
plot(x_e(1,1:end-1), x_n(1,1:end-1), LineWidth=2);
legend('real data',' Kalman filter');

