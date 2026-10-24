%% elliptic_pentarnion_exact.m
% Exact high-resolution trajectory for the irreducible genus-one Pentarnion flow.
% Authors: Alireza Poshtkohi and Susanne Pumpluen

clear; clc; close all;
m = 1/2;
[K,~] = ellipke(m);
T = 4*K;
N = 200001;
t = linspace(0,T,N).';
[sn,cn,dn] = ellipj(t,m);
v = cn.*dn;
P = [sn, v, sn.^2, sn.*v, sn.^3];

w=P(:,1); x=P(:,2); y=P(:,3); z=P(:,4); u=P(:,5);
F = [x-z+w.*x+3*w.*z-3*x.*y, ...
     y-(3/2)*w-w.^2+w.*y, ...
     2*w.*x, ...
     1-3*w.^2+(3/2)*u.*w, ...
     3*w.*z];
dP = [v, sn.^3-(3/2)*sn, 2*sn.*v, ...
      1-3*sn.^2+(3/2)*sn.^4, 3*sn.^2.*v];
residual = max(abs(F-dP),[],'all');

fprintf('Period T = %.15g\n',T);
fprintf('Samples = %d\n',N);
fprintf('Maximum exact residual = %.3e\n',residual);

figure;
plot3(w,z,u,'LineWidth',0.8); grid on;
xlabel('w'); ylabel('z'); zlabel('u');
title('Irreducible genus-one Pentarnion orbit');

figure; hold on;
plot(t,P,'LineWidth',0.8); grid on;
xlabel('t'); ylabel('canonical component');
legend('w','x (i)','y (j)','z (k)','u (ell)','Location','best');
title('One exact period, m=1/2');
